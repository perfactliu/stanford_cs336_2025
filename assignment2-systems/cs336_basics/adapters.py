from __future__ import annotations

import os
from typing import IO, Any, BinaryIO, List, Tuple, Dict, Iterable, Iterator, Optional
import collections
from jaxtyping import Float, Int

import numpy.typing as npt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import regex as re
from .pretokenization_example import find_chunk_boundaries
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        # 创建权重参数 W，形状为 [out_features, in_features]
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 初始化权重: truncated normal with std = sqrt(2 / (in + out))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape [..., in_features]
        # self.W: shape [out_features, in_features]
        # result: shape [..., out_features]
        return x @ self.W.T  # Note: not W @ x.T, so we transpose W


def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"] | None,
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)

    # 加载预训练权重
    if weights is not None:
        state_dict = {"W": weights}
        linear.load_state_dict(state_dict)

    # 执行前向传播
    return linear(in_features)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # 初始化权重：truncated normal，均值0，标准差1，截断范围[-3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch_size, seq_len] or any shape with int IDs
        # 输出: 对应位置查表 -> [*, embedding_dim]
        return self.weight[token_ids]


def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"] | None,
        token_ids: Int[Tensor, "..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    # 创建 embedding 模块
    embedding = Embedding(vocab_size, d_model)

    # 加载给定的 embedding 权重
    if weights is not None:
        embedding.load_state_dict({"weight": weights})

    # 查表返回对应向量
    return embedding(token_ids)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()

        # 中间维度 ≈ 8/3 * d_model，向上取最接近的64倍数
        d_ff = int((8 / 3) * d_model)
        d_ff = (d_ff + 63) // 64 * 64  # 向上取整到64的倍数

        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        # W1 和 W2 是 GLU 两个分支的投影层
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

        # 输出层将 GLU 的结果映射回 d_model
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x_proj = self.w3(x)  # -> [batch, seq, d_ff]
        x_gate = torch.sigmoid(self.w1(x))  # -> [batch, seq, d_ff]

        x_swiglu = x_proj * x_gate  # GLU with sigmoid gate
        return self.w2(x_swiglu)  # -> [batch, seq, d_model]


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"] | None,
        w2_weight: Float[Tensor, " d_model d_ff"] | None,
        w3_weight: Float[Tensor, " d_ff d_model"] | None,
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model)
    swiglu.d_ff = d_ff
    if w1_weight is not None:
        swiglu.w1.weight.data = w1_weight
        swiglu.w2.weight.data = w2_weight
        swiglu.w3.weight.data = w3_weight

    return swiglu(in_features)


def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]

    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=Q.dtype, device=Q.device))  # (..., queries, keys)

    if mask is not None:
        # If mask is True where we want to keep (i.e. attend), and False where to ignore:
        # We convert it so False becomes -inf
        scores = scores.masked_fill(~mask, float('-inf'))

    attn_weights = run_softmax(scores, dim=-1)  # (..., queries, keys)

    output = torch.matmul(attn_weights, V)  # (..., queries, d_v)

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(in_features=d_model, out_features=self.d_k)
        self.k_proj = Linear(in_features=d_model, out_features=self.d_k)
        self.v_proj = Linear(in_features=d_model, out_features=self.d_v)
        self.o_proj = Linear(in_features=self.d_v, out_features=d_model)

    def forward(self, in_features):
        seq_len = in_features.shape[-2]
        batch_shape = in_features.shape[:-2]
        # Flatten batch dims
        x = in_features.reshape(-1, seq_len, self.d_model)  # [B, S, D]

        # Project Q, K, V: shape [B, S, d_k * H]
        Q = self.q_proj(x).reshape(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, S, d_k]
        K = self.k_proj(x).reshape(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).reshape(-1, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, S, S]

        # Apply causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()  # [S, S]
        scores = scores.masked_fill(~mask, float('-inf'))

        # Attention weights
        attn = run_softmax(scores, dim=-1)  # [B, H, S, S]

        # Weighted sum
        out = torch.matmul(attn, V)  # [B, H, S, d_v]

        # Merge heads
        out = out.transpose(1, 2).reshape(-1, seq_len, self.num_heads * self.d_v)  # [B, S, D]

        # Output projection
        out = self.o_proj(out)  # [B, S, D]

        # Reshape back to original batch shape
        final_shape = (*batch_shape, seq_len, self.d_model)
        return out.reshape(final_shape)


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_k d_in"] | None,
        k_proj_weight: Float[Tensor, " d_k d_in"] | None,
        v_proj_weight: Float[Tensor, " d_v d_in"] | None,
        o_proj_weight: Float[Tensor, " d_model d_v"] | None,
        in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi_head_self_attention = MultiHeadSelfAttention(d_model, num_heads)

    if q_proj_weight is not None:
        multi_head_self_attention.q_proj.weight.data = q_proj_weight
        multi_head_self_attention.k_proj.weight.data = k_proj_weight
        multi_head_self_attention.v_proj.weight.data = v_proj_weight
        multi_head_self_attention.o_proj.weight.data = o_proj_weight

    return multi_head_self_attention(in_features)


class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(in_features=d_model, out_features=d_model)
        self.k_proj = Linear(in_features=d_model, out_features=d_model)
        self.v_proj = Linear(in_features=d_model, out_features=d_model)
        self.o_proj = Linear(in_features=d_model, out_features=d_model)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def forward(self, in_features, token_positions):
        seq_len = in_features.shape[-2]
        batch_shape = in_features.shape[:-2]
        # Flatten batch dims
        x = in_features.reshape(-1, seq_len, self.d_model)  # [B, S, D]
        B = x.shape[0]

        # Project Q, K, V: shape [B, S, d_k * H]
        Q = self.q_proj(x).reshape(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, S, d_k]
        K = self.k_proj(x).reshape(-1, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).reshape(-1, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # If token_positions is provided, apply RoPE to Q and K
        if token_positions is not None:
            # Expand token_positions to shape [B, H, S]
            token_positions = token_positions.reshape(-1, seq_len)  # [B, S]
            token_positions = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)  # [B, H, S]

        else:
            batch_size, seq_len, _ = in_features.shape
            position_ids = torch.arange(seq_len, device=in_features.device)  # [seq_len]
            token_positions = position_ids.view(1, 1, seq_len).expand(batch_size, self.num_heads, seq_len)  # [B, H, S]

        # Flatten for run_rope call
        Q = Q.reshape(-1, seq_len, self.d_k)
        K = K.reshape(-1, seq_len, self.d_k)
        pos = token_positions.reshape(-1, seq_len)

        Q = self.rope(Q, pos)
        K = self.rope(K, pos)

        # Restore shape to [B, H, S, d_k]
        Q = Q.reshape(B, self.num_heads, seq_len, self.d_k)
        K = K.reshape(B, self.num_heads, seq_len, self.d_k)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, S, S]

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device)).bool()
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Attention softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, S, S]

        # Apply attention weights to V
        out = torch.matmul(attn_weights, V)  # [B, H, S, d_v]

        # Merge heads: [B, S, H * d_v]
        out = out.transpose(1, 2).reshape(B, seq_len, self.num_heads * self.d_v)

        # Final linear projection
        out = self.o_proj(out)  # [B, S, D]

        # Reshape back to original batch shape
        return out.reshape(*batch_shape, seq_len, self.d_model)


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"] | None,
        k_proj_weight: Float[Tensor, " d_k d_in"] | None,
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi_head_self_attention_with_rope = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta)

    if q_proj_weight is not None:
        multi_head_self_attention_with_rope.q_proj.weight.data = q_proj_weight
        multi_head_self_attention_with_rope.k_proj.weight.data = k_proj_weight
        multi_head_self_attention_with_rope.v_proj.weight.data = v_proj_weight
        multi_head_self_attention_with_rope.o_proj.weight.data = o_proj_weight

    return multi_head_self_attention_with_rope(in_features, token_positions)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0, "d_k must be divisible by 2 for RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率项，RoPE中是 theta^{-2i/d_k}
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        # 生成 [max_seq_len, d_k//2]
        pos = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [max_seq_len, d_k//2]

        # 提前算好 cos 和 sin 缓存，并注册为 buffer
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)  # [max_seq_len, d_k//2]
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)  # [max_seq_len, d_k//2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        return: (..., seq_len, d_k)
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected last dim = {self.d_k}, got {d_k}"

        # 拆成偶数维和奇数维
        x_1, x_2 = x[..., ::2], x[..., 1::2]  # (..., seq_len, d_k/2)

        # 提取当前位置对应的 cos/sin 值
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        # RoPE核心旋转公式
        rotated_x = torch.stack([
            x_1 * cos - x_2 * sin,
            x_1 * sin + x_2 * cos
        ], dim=-1)  # (..., seq_len, d_k/2, 2)

        return rotated_x.flatten(-2)  # (..., seq_len, d_k)


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.norm1 = RMSNorm(d_model=d_model, eps=1e-5)
        self.norm2 = RMSNorm(d_model=d_model, eps=1e-5)
        self.multi_head_self_attention_with_rope = MultiHeadSelfAttentionWithRope(d_model, num_heads, max_seq_len, theta)
        self.swiglu = SwiGLU(d_model)
        if d_ff is not None:
            self.swiglu.d_ff = d_ff

    def forward(self, in_features, token_positions):
        # --------------------- Step 1: LayerNorm / RMSNorm ---------------------
        x = in_features
        x_norm1 = self.norm1(x)  # [B, S, D]

        # --------------------- Step 2: Multi-Head Attention with RoPE ---------------------
        out = self.multi_head_self_attention_with_rope(x_norm1, token_positions)

        x = x + out  # Residual 1

        # --------------------- Step 3: RMSNorm ---------------------
        x_norm2 = self.norm2(x)   # [B, S, D]

        # --------------------- Step 4: Feed-Forward Network (SwiGLU-like) ---------------------
        x_ff = self.swiglu(x_norm2)

        return x + x_ff  # Residual 2


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)

    transformer_block.norm1.load_state_dict({"weight": weights["ln1.weight"]})
    transformer_block.multi_head_self_attention_with_rope.q_proj.weight.data = weights["attn.q_proj.weight"]
    transformer_block.multi_head_self_attention_with_rope.k_proj.weight.data = weights["attn.k_proj.weight"]
    transformer_block.multi_head_self_attention_with_rope.v_proj.weight.data = weights["attn.v_proj.weight"]
    transformer_block.multi_head_self_attention_with_rope.o_proj.weight.data = weights["attn.output_proj.weight"]
    transformer_block.norm2.load_state_dict({"weight": weights["ln2.weight"]})
    transformer_block.swiglu.w1.weight.data = weights["ffn.w1.weight"]
    transformer_block.swiglu.w2.weight.data = weights["ffn.w2.weight"]
    transformer_block.swiglu.w3.weight.data = weights["ffn.w3.weight"]

    return transformer_block(in_features, None)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_len = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model=d_model, eps=1e-5)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.linear = Linear(d_model, vocab_size)

    def forward(self, in_indices):
        # === 1. Embedding lookup ===
        token_embeds = self.embedding(in_indices)  # [B, S, D]

        # === 2. Transformer blocks ===
        x = token_embeds

        for block in self.blocks:
            x = block(x, None)

        # === 3. Final RMSNorm ===
        x = self.norm(x)  # [B, S, D]

        # === 4. Language Modeling Head ===
        logits = self.linear(x)

        return logits


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    transformer_lm.embedding.load_state_dict({"weight": weights["token_embeddings.weight"]})
    for num_layer, block in enumerate(transformer_lm.blocks):
        block.norm1.load_state_dict({"weight": weights[f"layers.{num_layer}.ln1.weight"]})
        block.multi_head_self_attention_with_rope.q_proj.weight.data = weights[f"layers.{num_layer}.attn.q_proj.weight"]
        block.multi_head_self_attention_with_rope.k_proj.weight.data = weights[f"layers.{num_layer}.attn.k_proj.weight"]
        block.multi_head_self_attention_with_rope.v_proj.weight.data = weights[f"layers.{num_layer}.attn.v_proj.weight"]
        block.multi_head_self_attention_with_rope.o_proj.weight.data = weights[f"layers.{num_layer}.attn.output_proj.weight"]
        block.norm2.load_state_dict({"weight": weights[f"layers.{num_layer}.ln2.weight"]})
        block.swiglu.w1.weight.data = weights[f"layers.{num_layer}.ffn.w1.weight"]
        block.swiglu.w2.weight.data = weights[f"layers.{num_layer}.ffn.w2.weight"]
        block.swiglu.w3.weight.data = weights[f"layers.{num_layer}.ffn.w3.weight"]
    transformer_lm.norm.load_state_dict({"weight": weights[f"ln_final.weight"]})
    transformer_lm.linear.load_state_dict({"weight": weights["lm_head.weight"]})

    return transformer_lm(in_indices)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {'device': device, 'dtype': dtype}

        # 可学习缩放参数 γ（与 LayerNorm 相似）
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute RMS across last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)  # shape: (..., 1)

        # Normalize and scale
        normed = x / rms  # (..., d_model)
        output = normed * self.weight  # learnable scale

        return output.to(orig_dtype)


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, "d_model"] | None,
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    # 创建 RMSNorm 模块实例
    rmsnorm = RMSNorm(d_model=d_model, eps=eps)

    if weights is not None:
        # 加载给定权重
        rmsnorm.load_state_dict({"weight": weights})

    # 前向传播
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 样本的起点最大位置是 len(dataset) - context_length - 1
    max_start = len(dataset) - context_length - 1
    assert max_start >= 0, "Dataset too small for given context_length"

    # 随机选择起始位置（batch_size 个）
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # 用列表构造输入和标签对
    inputs = np.stack([dataset[i: i + context_length] for i in starts])
    targets = np.stack([dataset[i + 1: i + 1 + context_length] for i in starts])

    # 转换为 torch.LongTensor 并放到指定 device
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_vals, _ = in_features.max(dim=dim, keepdim=True)
    shifted = in_features - max_vals
    exp_tensor = torch.exp(shifted)
    sum_exp = exp_tensor.sum(dim=dim, keepdim=True)
    return exp_tensor / sum_exp


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[
    Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Step 1: subtract max for numerical stability
    stable_inputs = inputs - inputs.max(dim=-1, keepdim=True).values  # shape: [batch_size, vocab_size]

    # Step 2: compute logsumexp along vocab dimension
    logsumexp = torch.log(torch.sum(torch.exp(stable_inputs), dim=-1))  # shape: [batch_size]

    # Step 3: pick the logit corresponding to the correct class
    correct_logits = stable_inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # shape: [batch_size]

    # Step 4: compute cross-entropy: -correct_logit + logsumexp
    losses = -correct_logits + logsumexp  # shape: [batch_size]

    # Step 5: average over batch
    return losses.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    params = [p for p in parameters if p.grad is not None]

    # Compute global L2 norm across all gradients
    total_norm_sq = sum(p.grad.data.norm(2).item() ** 2 for p in params)
    total_norm = total_norm_sq ** 0.5

    # If norm exceeds max, rescale all gradients in-place
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            p.grad.data.mul_(scale)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    class AdamW(Optimizer):
        def __init__(
                self,
                params,
                lr: float = 3e-4,
                betas: tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8,
                weight_decay: float = 0.01,
        ):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        def step(self, closure: Optional[callable] = None):
            loss = closure() if closure is not None else None

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    state = self.state[p]

                    # Initialize state
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)  # m_t
                        state["exp_avg_sq"] = torch.zeros_like(p.data)  # v_t

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    step = state["step"]

                    # Weight decay (AdamW decouples it from gradient)
                    if group["weight_decay"] != 0:
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            return loss

    return AdamW


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Linear warmup: interpolate from min_lr to max_lr
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (it / warmup_iters)
    elif it < warmup_iters + cosine_cycle_iters:
        # Cosine decay
        progress = (it - warmup_iters) / cosine_cycle_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        # After decay period, learning rate stays at min_learning_rate
        return min_learning_rate


def run_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.id_to_token = {i: token for i, token in vocab.items()}
        self.token_to_id = {token: i for i, token in vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # Handle special tokens
        self.special_tokens = []
        if special_tokens:
            for token in special_tokens:
                btoken = token.encode('utf-8')
                if btoken not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[btoken] = idx
                    self.id_to_token[idx] = btoken
                self.special_tokens.append(btoken)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        vocab = {}
        with open(vocab_filepath, 'rb') as f:
            for line in f:
                id_str, token = line.strip().split()
                vocab[int(id_str)] = token

        merges = []
        with open(merges_filepath, 'rb') as f:
            for line in f:
                a, b = line.strip().split()
                merges.append((a, b))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        # Short-circuit: entire string is a special token
        if text.encode("utf-8") in self.special_tokens:
            return [self.token_to_id[text.encode("utf-8")]]

        # Step 1: 特殊 token 切分（保留 byte token）
        special_token_set = set(self.special_tokens)
        pattern = "|".join(re.escape(tok.decode("utf-8")) for tok in special_token_set)
        parts = re.split(f"({pattern})", text)

        tokens = []

        for part in parts:
            if not part:
                continue
            bpart = part.encode("utf-8")
            if bpart in special_token_set:
                tokens.append(bpart)  # 直接保留
            else:
                # 字节切分为单字节 byte
                token_bytes = [bytes([b]) for b in bpart]

                # BPE merge
                while len(token_bytes) >= 2:
                    pairs = [(token_bytes[i], token_bytes[i + 1]) for i in range(len(token_bytes) - 1)]
                    pair_ranks = [(pair, self.bpe_ranks.get(pair, float('inf'))) for pair in pairs]
                    best_pair, best_rank = min(pair_ranks, key=lambda x: x[1], default=(None, float('inf')))
                    if best_rank == float('inf'):
                        break
                    merged_token = best_pair[0] + best_pair[1]
                    new_tokens = []
                    i = 0
                    while i < len(token_bytes):
                        if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i + 1]) == best_pair:
                            new_tokens.append(merged_token)
                            i += 2
                        else:
                            new_tokens.append(token_bytes[i])
                            i += 1
                    token_bytes = new_tokens
                tokens.extend(token_bytes)

        # Map to ids
        unk_id = self.token_to_id.get(b"<unk>", 0)
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        byte_tokens = []

        for idx in ids:
            tok = self.id_to_token.get(idx)
            if tok is None:
                # 若遇到未知ID，用 <unk> 表示
                byte_tokens.append(b"<unk>")
            else:
                byte_tokens.append(tok)

        # 拼接所有 byte token
        raw_bytes = b''.join(byte_tokens)

        # 解码为 utf-8 文本
        try:
            return raw_bytes.decode('utf-8', errors='replace')
        except Exception as e:
            print("decode error:", e)
            return raw_bytes.decode('utf-8', errors='ignore')


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    escaped_specials = [re.escape(t) for t in special_tokens]
    split_pattern = "|".join(escaped_specials)
    splitter = re.compile(split_pattern)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 预分词阶段
    all_tokenized = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 8, "<|endoftext|>".encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 去除special tokens
            segments = splitter.split(chunk)
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                # regex 分词
                tokens = [match.group(0) for match in re.finditer(PAT, segment)]
                all_tokenized.append(tokens)

    corpus = []
    for line in all_tokenized:
        for token in line:
            token_bytes = token.encode("utf-8")
            corpus.append(tuple([bytes([b]) for b in token_bytes]))

    # 初始化 vocab & merges
    merges: List[Tuple[bytes, bytes]] = []
    token_to_id: Dict[bytes, int] = {}

    # 初始化 vocab 为所有单字节字符 + 特殊 token + 出现在语料中的 byte
    byte_vocab = {bytes([i]) for i in range(256)}
    byte_vocab |= {b for word in corpus for b in word}
    byte_vocab |= {t.encode("utf-8") for t in special_tokens}

    vocab_list = sorted(byte_vocab)
    token_to_id = {token: i for i, token in enumerate(vocab_list)}

    # Merge loop
    while len(token_to_id) < vocab_size:
        print(len(token_to_id))
        pair_freq = collections.Counter()

        # 统计 pair 频率
        for word in corpus:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += 1

        if not pair_freq:
            break

        # 找出最高频率
        max_freq = max(pair_freq.values())
        # 找出所有具有最高频率的 pair
        candidates = [pair for pair, freq in pair_freq.items() if freq == max_freq]
        # 从中选出字典序最大的那个
        most_common = max(candidates)
        merges.append(most_common)

        # 执行合并（更新所有词）
        new_corpus = []
        merged_token = most_common[0] + most_common[1]
        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if (
                        i < len(word) - 1
                        and word[i] == most_common[0]
                        and word[i + 1] == most_common[1]
                ):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(tuple(new_word))
        corpus = new_corpus

        # 加入新 token 到 vocab
        if merged_token not in token_to_id:
            token_to_id[merged_token] = len(token_to_id)

    # 返回 vocab 和 merges（按 token ID 排序）
    id_to_token = {v: k for k, v in token_to_id.items()}
    return id_to_token, merges
