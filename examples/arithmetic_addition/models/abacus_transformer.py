"""Decoder-only Transformer with sinusoidal Abacus embeddings.

Reference: "Transformers Can Do Arithmetic with the Right Embeddings"
(McLeish et al., 2024). https://arxiv.org/abs/2405.17399

The paper uses a *learned* embedding table indexed by digit column.  Because
that table is an absolute positional embedding, it cannot generalize to column
indices unseen during training: with 2-6 digit operands (max answer length 7)
the row for column 8 is never trained, which produces a sharp accuracy cliff at
8-digit operands.

This implementation keeps the Abacus *column id* (position relative to the
start of each number, units digit first because the data is LSB-first) but
replaces the learned lookup with a deterministic sinusoidal function of the
column index.  Every column, seen or unseen, therefore receives a unique and
smoothly related vector, so digit alignment can generalize beyond the training
length.  A small learned linear projection mixes the fixed Fourier basis, and a
single learned vector represents all non-digit tokens:

1. **Sinusoidal column encoding.**  ``[sin(c * f), cos(c * f)]`` for the same
   frequencies as the classic Transformer, where ``c`` is the one-based column
   id and ``f`` is a fixed geometric frequency band.  The result is projected
   to the model width.  Non-digit tokens use one shared learned vector.

2. **Random training offset.**  As in the paper, a single offset
   ``k ~ U[0, max_offset]`` is sampled per batch and added to every digit
   column during training; inference uses ``k = 0``.  With a functional
   encoding the offset is a phase shift rather than a table lookup, so it is
   handled naturally and cannot create untrained rows.

3. **Addition, not concatenation.**  The Abacus embedding is added to the token
   embedding, using the same width as the baseline's hidden size.

The LSB-first data format is required: it makes the column id of an
already-generated token independent of future digits, so teacher-forced
training and autoregressive inference assign identical column ids.  The local
dataset (``dataset_impl.arithmetic_addition``) uses this format for both
experiments.

Out of scope for this experiment (paper add-ons, not part of the core Abacus
embedding change): input injection, progressive loss, and depth-recurrent
(looped) layers.  Keeping the same depth/width/budget as the RoPE baseline
isolates the effect of the Abacus positional encoding.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from models.causal_lm import CausalLMCollateFn, CausalLMCrossEntropyLoss

NON_DIGIT_INDEX = 0


def compute_column_ids(
    input_ids: torch.Tensor,
    digit_token_start: int,
    num_digit_tokens: int,
) -> torch.Tensor:
    """Return the Abacus column id of every token.

    Scanning each sequence from left to right, the counter is incremented for
    every digit and reset to zero at every non-digit.  Because inputs are
    LSB-first, the first digit of each number (its units digit) receives column
    id 1, the next one 2, and so on.  Non-digit tokens receive the shared
    non-digit slot ``0``.

    This is the same span-local, one-based numbering as the paper's
    ``Abacus.helper``.  It is independent of left/right padding because pad
    tokens are non-digits.

    Args:
        input_ids: ``(batch, seq_len)`` token ids.
        digit_token_start: first token id belonging to the digit block.
        num_digit_tokens: size of the contiguous digit token block.

    Returns:
        ``(batch, seq_len)`` long tensor.  Values are 0 for non-digits and
        ``1..L`` for digits, where ``L`` is the digit's position from the start
        of its number.
    """
    B, S = input_ids.shape
    is_digit = (input_ids >= digit_token_start) & (input_ids < digit_token_start + num_digit_tokens)
    column_ids = torch.zeros((B, S), dtype=torch.long, device=input_ids.device)
    counter = torch.zeros(B, dtype=torch.long, device=input_ids.device)
    for t in range(S):
        digit_t = is_digit[:, t]
        counter = torch.where(digit_t, counter + 1, torch.zeros_like(counter))
        column_ids[:, t] = torch.where(digit_t, counter, torch.zeros_like(counter))
    return column_ids


class AbacusEmbedding(nn.Module):
    """Sinusoidal (functional) encoding indexed by digit column.

    Unlike the paper's learned table, the digit-column embedding is a fixed
    Fourier basis evaluated at the column index, so columns never seen during
    training still receive a well-defined, smoothly related vector.  A learned
    projection mixes the basis, and one learned vector is shared by all
    non-digit tokens (BOS, ``+``, ``=``, EOS, padding).
    """

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int = 1024,
        max_offset: int = 100,
        digit_token_start: int = 5,
        num_digit_tokens: int = 10,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sinusoidal Abacus encoding")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if max_offset < 0:
            raise ValueError("max_offset must be non-negative")
        if num_digit_tokens <= 0 or digit_token_start < 0:
            raise ValueError("num_digit_tokens must be positive and digit_token_start non-negative")

        half = embedding_dim // 2
        # Same geometric frequency bands as the classic sinusoidal encoding.
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        # Learned mixing of the fixed basis; keeps the encoding full-rank.
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # Shared learned vector for every non-digit token (column id 0).
        self.non_digit = nn.Parameter(torch.zeros(embedding_dim))

        # Kept for API/config compatibility; the functional encoding has no
        # table size limit, so max_seq_len is only validated, never indexed.
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.max_offset = max_offset
        self.digit_token_start = digit_token_start
        self.num_digit_tokens = num_digit_tokens

    def forward(self, column_ids: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Map column ids ``(B, S)`` to embeddings ``(B, S, embedding_dim)``.

        ``offset`` is added only to digit positions (>0), matching the paper's
        ``output[output > 0] += k``.  The non-digit slot is never shifted.
        """
        ids = column_ids.to(torch.float32)
        if offset:
            ids = torch.where(column_ids > 0, ids + offset, ids)
        args = ids.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = self.proj(emb)
        is_digit = (column_ids > 0).unsqueeze(-1)
        return torch.where(is_digit, emb, self.non_digit.to(emb.dtype))


class Attention(nn.Module):
    """Causal multi-head self-attention without positional encoding.

    Positions are injected by the Abacus embeddings added before the block
    stack, so attention itself is position-agnostic.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, C = x.shape

        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if padding_mask is not None:
            causal_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            mask = causal_mask & padding_mask
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=False,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        return self.dropout(self.out(context))


class TransformerBlock(nn.Module):
    """Pre-LN decoder block: causal self-attention + feedforward network."""

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), padding_mask=padding_mask)
        x = x + self.feedforward(self.norm2(x))
        return x


class AbacusTransformerModel(nn.Module):
    """Decoder-only Transformer with sinusoidal Abacus positional embeddings.

    Architecture:
        token ids -> token embedding (scale by sqrt(hidden_size))
                  -> + sinusoidal Abacus embedding of the digit column
                  -> N x TransformerBlock (standard causal self-attention + FFN)
                  -> LayerNorm -> linear projection to vocab logits

    The positional signal is the only intentional architectural difference from
    the RoPE baseline.  The Abacus depth is configurable per experiment (the
    local SinAbacus run uses 8 layers); width, heads, feedforward width,
    dropout, optimizer, data, and training schedule match the baseline.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        max_offset: int = 100,
        digit_token_start: int = 5,
        num_digit_tokens: int = 10,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.abacus = AbacusEmbedding(
            embedding_dim=hidden_size,
            max_seq_len=max_seq_len,
            max_offset=max_offset,
            digit_token_start=digit_token_start,
            num_digit_tokens=num_digit_tokens,
        )
        self.digit_token_start = digit_token_start
        self.num_digit_tokens = num_digit_tokens
        self.max_offset = max_offset
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.norm = nn.LayerNorm(hidden_size)

    def compute_column_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Base (offset-free) Abacus column ids for tests and inspection."""
        return compute_column_ids(
            input_ids,
            digit_token_start=self.digit_token_start,
            num_digit_tokens=self.num_digit_tokens,
        )

    def _sample_offset(self, device: torch.device) -> int:
        """Sample the paper's per-batch random training offset.

        The same offset is used for every digit in the batch and is zero during
        evaluation (``self.training`` is False).
        """
        if not self.training or self.max_offset <= 0:
            return 0
        return int(torch.randint(0, self.max_offset + 1, (1,), device=device).item())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S = input_ids.shape
        token_embeddings = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)

        column_ids = self.compute_column_ids(input_ids)
        abacus_embeddings = self.abacus(column_ids, offset=self._sample_offset(input_ids.device))
        x = token_embeddings + abacus_embeddings

        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if not attention_mask.all().item():
                padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)
        elif (input_ids == self.embedding.padding_idx).any().item():
            padding_mask = (input_ids != self.embedding.padding_idx).reshape(B, 1, 1, S)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.norm(x)
        return self.proj(x)


__all__ = [
    "NON_DIGIT_INDEX",
    "compute_column_ids",
    "AbacusEmbedding",
    "Attention",
    "TransformerBlock",
    "AbacusTransformerModel",
    "CausalLMCollateFn",
    "CausalLMCrossEntropyLoss",
]
