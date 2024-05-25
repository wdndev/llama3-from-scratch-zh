import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """ Important parameters of our model """

    dim: int = 4096 # input embedding dimension
    n_layers: int = 32 # number of times the transformer block is repeated
    n_heads: int = 32 # number of heads for Queries
    n_kv_heads: Optional[int] = None # number of heads for Keys and Values
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5 # small value added to the denominator in RMSNorm for numerical stability
    rope_theta: float = 500000 # scaling factor for frequency computation in RoPE

    max_batch_size: int = 32 # needed for KV cache
    max_seq_len: int = 2048 # needed for KV cache

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # gamma parameter
    
    def _norm(self, x: torch.tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # (B, seq_len, dim)

    def forward(self, x: torch.tensor):
        return self.weight * self._norm(x.float()).type_as(x) # (dim) * (B, seq_len, dim) --> (B, seq_len, dim)


def precompute_freqs_complex(head_size: int, seq_len: int, device: str, theta: float = 10000.0):
    """ Precomputing the frequency tensor with complex exponentials for the given sequence length and dimensions """

    assert head_size % 2 == 0, "embedding dimension must be even"
    # first computing the theta parameter according to the formula in paper
    # theta_i = 10000 ^ (-2 * (i-1)/d) for i = [1, 2, 3, .... , d/2]
    theta = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size)).to(device) # (head_size/2)
    # now computing the m (positions) paƒrameter
    m = torch.arange(seq_len, device=device) # (seq_len)
    # using outer product to multiply each theta with each m (positions)
    freqs = torch.outer(m, theta).float() # (seq_len) * (head_size/2) --> (seq_len, head_size/2)
    # now converting the above frequencies into complex numbers using the polar form c = R * exp(i * m * theta), where R = 1
    freqs_complex =  torch.polar(torch.ones_like(freqs), freqs) # (seq_len, head_size/2)

    return freqs_complex


def apply_rotary_position_embeddings(x: torch.tensor, freqs_complex: torch.tensor, device: str):
    """  Applying rotary position embeddings to input tensors using the given frequency tensor """

    # converting two consecutive numbers into a single complex number
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # (B, seq_len, H, head_size/2)
    # reshaping freqs_complex tensor to match the shape of x_complex tensor in order to perform element-wise operations
    # adding the batch and head dimensions
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2) # (seq_len, head_size/2) --> (1, seq_len, 1, head_size/2)
    x_rotated = x_complex * freqs_complex # (B, seq_len, H, head_size/2) * (1, seq_len, H, head_size/2) --> (B, seq_len, H, head_size/2)
    # converting the above rotated tensor into a new tensor where the first element is the real part and the second element is the imaginary part of the complex number
    x_out = torch.view_as_real(x_rotated) # (B, seq_len, H, head_size/2) --> (B, seq_len, H, head_size/2, 2)
    # flattening the above tensor
    x_out = x_out.reshape(*x.shape) # (B, seq_len, H, head_size/2, 2) --> (B, seq_len, H, head_size)
    
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.tensor, n_repeat: int):
    """ Repeating the heads of keys and values to match the number of query heads """

    B, seq_len_kv, n_kv_heads, head_size = x.shape
    if n_repeat == 1:
        return x
    else:
        return (
            x[:, :, :, None, :] # (B, seq_len, n_kv_heads, 1, head_size), added a new dimension
            .expand(B, seq_len_kv, n_kv_heads, n_repeat, head_size)
            .reshape(B, seq_len_kv, n_kv_heads * n_repeat, head_size)
        )
    

class Attention(nn.Module):
    """ Grouped-Query Attention using KV cache with RoPE applied to queries and keys """

    def __init__(self, head_size: int, params: ModelArgs):
        super().__init__()

        self.head_size = head_size
        # number of heads for keys and values
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        # number of heads for queries
        self.n_q_heads = params.n_heads
        # number of times keys and values should be repeated
        self.n_repeat = self.n_q_heads // self.n_kv_heads

        self.wq = nn.Linear(params.dim, self.n_q_heads * self.head_size, bias=False)
        self.wk = nn.Linear(params.dim, self.n_kv_heads * self.head_size, bias=False)
        self.wv = nn.Linear(params.dim, self.n_kv_heads * self.head_size, bias=False)
        self.wo = nn.Linear(params.n_heads * self.head_size, params.dim, bias=False)

        self.cache_k = torch.zeros((params.max_batch_size, params.max_seq_len, self.n_kv_heads, head_size))
        self.cache_v = torch.zeros((params.max_batch_size, params.max_seq_len, self.n_kv_heads, head_size))
        
    def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.tensor):
        B, seq_len, C = x.shape # (batch_size, 1, dim)

        xq = self.wq(x) # (B, 1, dim) --> (B, 1, n_q_heads * head_size)
        xk = self.wk(x) # (B, 1, dim) --> (B, 1, n_kv_heads * head_size)
        xv = self.wv(x) # (B, 1, dim) --> (B, 1, b_kv_heads * head_size)

        xq = xq.view(B, seq_len, self.n_q_heads, self.head_size) # (B, 1, n_q_heads * head_size) --> (B, 1, n_q_heads, head_size)
        xk = xk.view(B, seq_len, self.n_kv_heads, self.head_size) # (B, 1, n_kv_heads * head_size) --> (B, 1, n_kv_heads, head_size)
        xv = xv.view(B, seq_len, self.n_kv_heads, self.head_size) # (B, 1, n_kv_heads * head_size) --> (B, 1, n_kv_heads, head_size)

        xq = apply_rotary_position_embeddings(xq, freqs_complex, device=x.device) # (B, 1, n_q_heads, head_size)
        xk = apply_rotary_position_embeddings(xk, freqs_complex, device=x.device) # (B, 1, n_kv_heads, head_size)
        
        # replacing the entry for this token in the cache
        self.cache_k[:B, start_pos:start_pos+seq_len] = xk
        self.cache_v[:B, start_pos:start_pos+seq_len] = xv
        
        # retrieving all the cached keys and values so far
        keys = self.cache_k[:B, 0:start_pos+seq_len] # (B, seq_len_kv, n_kv_heads, head_size)
        values = self.cache_v[:B, 0:start_pos+seq_len] # (B, seq_len_kv, n_kv_heads, head_size)

        # repeating the heads of keys and values to match the number of query heads
        # repeating the heads is not the most optimal way but this is how META has done it too
        keys = repeat_kv(keys, self.n_repeat)
        values = repeat_kv(values, self.n_repeat)

        xq = xq.transpose(1, 2) # (B, 1, n_q_heads, head_size) --> (B, n_q_heads, 1, head_size)
        keys = keys.transpose(1, 2) # (B, seq_len_kv, n_q_heads, head_size) --> (B, n_q_heads, seq_len_kv, head_size)
        values = values.transpose(1, 2) # (B, seq_len_kv, n_q_heads, head_size) --> (B, n_q_heads, seq_len_kv, head_size)

        # computing attention scores
        scores = xq @ keys.transpose(2, 3) * self.head_size**-0.5 # (B, n_q_heads, 1, head_size) @ (B, n_q_heads, head_size, seq_len_kv) --> (B, n_q_heads, 1, seq_len_kv)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # taking weighted aggregation of values
        output = scores @ values # (B, n_q_heads, 1, seq_len_kv) @ (B, n_q_heads, seq_len_kv, head_size) --> (B, n_q_heads, 1, head_size)
        # concatenating outputs from all the heads
        output = (output.transpose(1, 2).contiguous().view(B, seq_len, -1)) # (B, n_q_heads, 1, head_size) --> (B, n_q_heads, head_size, 1) --> (B, 1, n_heads * head_size)
        output = self.wo(output) # (B, 1, n_heads * head_size) --> (B, 1, dim)

        return output


class FeedForward(nn.Module):
    """ Feed forward with SwiGLU """

    def __init__(self, params: ModelArgs):
        super().__init__()
        hidden_dim = 4 * params.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if params.ffn_dim_multiplier is not None:
            hidden_dim = int(params.ffn_dim_multiplier * hidden_dim)
        # rounding the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = params.multiple_of * ((hidden_dim + params.multiple_of - 1) // params.multiple_of)

        self.w1 = nn.Linear(params.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, params.dim, bias=False)
        self.w3 = nn.Linear(params.dim, hidden_dim, bias=False)

        def forward(self, x: torch.tensor):
            # in SwiGLU, the Swish function is used to gate the linear function of GLU
            # swish(x) = x * sigmoid(beta * x)
            # when beta = 1, swish function becomes same as the sigmoid linear unit function (SiLU)
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, head_size: int, params: ModelArgs):
        super().__init__()

        self.head_size = head_size
        self.params = params
        self.n_heads = params.n_heads
        self.attention = Attention(head_size, params)
        self.ffwd = FeedForward(params)

        # normalization before self attention
        self.attention_norm = RMSNorm(params.dim, eps=params.norm_eps)
        # normalization before the feed forward block
        self.ffwd_norm = RMSNorm(params.dim, eps=params.norm_eps)

        def forward(self, x: torch.tensor, start_pos: int, freqs_complex: torch.tensor):
            # residual connections and root mean square layer normalization
            h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex) # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
            output = h + self.ffwd.forward(self.ffwd_norm(h))  # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)

            return output
        

class Transformer(nn.Module):
    """ Transformer module """

    def __init__(self, params = ModelArgs):
        super().__init__()

        assert params.vocab_size != -1, "vocab must be set, cannot be equal to -1"

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.head_size = self.params.dim // self.params.n_heads

        # interspersing communcation and computation by replicating the transformer block sequentially  
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # final normalization layer
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # final language model head
        self.output = nn.Linear(params.dim, self.vocab_size)

        # note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096
        # adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning
        self.freqs_complex = precompute_freqs_complex(self.head_size, params.max_seq_len * 2, device=self.params.device, theta=params.rope_theta)

    def forward(self, tokens: torch.tensor, start_pos: int):
        # tokens is a (batch_size (B), sequence_length (seq_len)) tensor of integers
        B, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"

        x = self.tok_embeddings(tokens) # (B, seq_len) --> (B, seq_len, dim)

        # retrieving the pairs (m, theta) corresponding to the positions [start_pos, start_pos + sequence_length]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # sequentially applying all the transformer blocks (decoder layers)
        for layer in self.layers:
            h = self.layers(x, start_pos, freqs_complex) # (B, seq_len , dim)
        h = self.norm(x) # (B, seq_len , dim)
        logits = self.output(h).float() # (B, seq_len, vocab_size)

        return logits
    