import torch
import time
import json
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from model import ModelArgs, Transformer
from tokenizer import Tokenizer


class Llama:
    """ Llama class """

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        """ Building a Llama instance by initializing and loading a model checkpoint """

        # loading the model
        if load_model:
            # loading the checkpoint
            start_time = time.time()
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            checkpoint_path = checkpoints[0]
            print(f"Loading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - start_time):.2f}s")
        
        start_time = time.time()
        # building the parameters (model arguments)
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
            )

        # loading the tokenizer
        tokenizer = Tokenizer(tokenizer_path)
        # using the tokenizer to populate the vocabulary size of our model
        model_args.vocab_size = tokenizer.n_words

        # setting the default torch tensor type 
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model = Transformer(model_args).to(device)
        if load_model:
            # since we are computing RoPE during inference, we don't need the frequencies provided by META
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - start_time):.2f}s")

        return Llama(model, model_args, tokenizer)
    
    def _sample_top_p(self, probs, p):
        """ Performing top-p (nucleus) sampling on a given probability distribution """

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1) # (B, vocab_size)
        # substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
        mask = probs_sum - probs_sort > p # (B, vocab_size)
        # zeroing out all the probabilities of tokens that are not selected by the top p
        probs_sort[mask] = 0.0 
        # redistributing the probabilities so that they sum up to 1
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # sampling a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # gathering the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token

    @torch.inference_mode()
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        """ Generating new tokens and performing text completion for a list of prompts """

        params = self.model.params

        # if maximum length of the generated completion sequence is not provided, it's set to the model's maximum sequence length minus 1
        if max_gen_len is None:
            max_gen_len = params.max_seq_len - 1

        # making sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= params.max_batch_size, f"batch size must be less than or equal to {params.max_batch_size}"

        # converting each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
        # making sure the maximum prompt length is not larger than the maximum sequence length
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len, f"prompt length must be less than or equal to {params.max_seq_len}"
        # total number of tokens we want to generate using the model
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # creating a list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # populating the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        
        # variable that tell us if we reached the end of sentence in any of the prompts
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # True if the token is a prompt token, False otherwise
        prompt_tokens_mask = tokens != pad_id

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # applying temperature before softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # greedily selecting the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # replacing the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # EOS is reached only if we find a stop token in a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            if all(eos_reached):
                break

        out_tokens, out_text = [], []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cutting to max gen len
            start = len(prompt_tokens[prompt_index])
            current_prompt_tokens = current_prompt_tokens[start : len(prompt_tokens[prompt_index]) + max_gen_len]
            probs = None

            # cutting to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                eos_idx = current_prompt_tokens.index(stop_token)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]

            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)
    

# inferencing the model
if __name__ == "__main__":
    torch.manual_seed(42)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available and allow_cuda else "cpu"

    prompts = [
        "In a nutshell, self attention is the ",
        "If there were 30 hours in a day, then",

        # few shot prompt
        """Translate English to French:

        clouds -> des nuages
        ice cream -> glace
        dog -> chien
        cheese ->""",

        # zero shot prompt
        """Do you think the following person is the best batsman in the world?
        Name: Virat Kohli
        Answer: 
        """
    ]

    model = Llama.build(
        checkpoints_dir="Meta-Llama-3-8B/",
        tokenizer_path="Meta-Llama-3-8B/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)