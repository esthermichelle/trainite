"""Decoder-only Transformer with Abacus positional embeddings and a looped block stack.

Reference: "Transformers Can Do Arithmetic with the Right Embeddings"
(McLeish et al., 2024). https://arxiv.org/abs/2405.17399

Two training-time changes force the network to follow the digit-column structure
of arithmetic instead of the absolute sequence position:

1. **Abacus embeddings** (Section 2.1): every digit token receives a positional
   signal indexed by its *column* — its distance from the right end of the
   number it belongs to.  The rightmost digit of every number (operand or
   result) is column 0, so numbers of different lengths are aligned "as on an
   abacus".  Non-digit tokens (``+``, ``<SEP>``, ``<BOS>``, pad) share one
   dedicated slot.  The embedding is a *fixed* sinusoidal table, not learned.
   Index lookup is cyclic (``column %% R``) so numbers longer than ``R``
   columns wrap around and reuse the same embeddings.

2. **Looped transformer** (Section 2.2): the same block stack is applied
   ``num_loops`` times (recurrent reuse of weights).  This adds effective depth
   without adding parameters, which the paper found important for handling
   carries and longer computations.

At inference, a number longer than anything seen in training still maps its
digits to columns 0..R-1 (already seen during training), so the model
extrapolates in operand length.  The token-level column index is recomputed on
every forward pass directly from ``input_ids`` — no dataset, collate or trainer
changes are needed.
"""

import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class AbacusPositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal table for digit-column positions.

    Row ``i`` in ``[0, R-1]`` is the embedding of digit column ``i`` (distance
    from the right end of a number).  Row ``R`` is a single shared slot for all
    non-digit tokens.  Generated once from the standard sinusoidal formula and
    stored as a non-persistent buffer, so it does not participate in
    checkpointing or backpropagation.
    """

    def __init__(self, abacus_dim: int, num_cyclic_positions: int) -> None:
        super().__init__()
        if abacus_dim % 2 != 0:
            raise ValueError(f"abacus_dim must be even, got {abacus_dim}.")
        if num_cyclic_positions <= 0:
            raise ValueError(f"num_cyclic_positions must be positive, got {num_cyclic_positions}.")
        self.num_cyclic_positions = num_cyclic_positions

        num_rows = num_cyclic_positions + 1  # extra row for non-digit tokens
        position = torch.arange(num_rows, dtype=torch.float32).unsqueeze(1)
        # Same frequency divider as the original Transformer sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, abacus_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / abacus_dim)
        )
        table = torch.zeros(num_rows, abacus_dim)
        table[:, 0::2] = torch.sin(position * div_term)
        half = abacus_dim // 2
        table[:, 1::2] = torch.cos(position[:half] * div_term[:half])
        self.register_buffer("table", table, persistent=False)

    def forward(self, column_ids: torch.Tensor) -> torch.Tensor:
        """Map integer column ids (B, S) to embeddings (B, S, abacus_dim)."""
        return self.table[column_ids]


def compute_column_ids(
    input_ids: torch.Tensor,
    digit_token_start: int,
    num_digit_tokens: int,
    num_cyclic_positions: int,
) -> torch.Tensor:
    """Recover the digit-column index of every token from raw token ids.

    Scanning each sequence from right to left, a counter tracks the distance to
    the most recent non-digit token: the first digit counted from the right
    (i.e. the units digit of the number it belongs to) gets column 0, the next
    digit to the left column 1, and any non-digit token resets the counter and
    itself receives the dedicated non-digit slot ``R``.  This is padding-order
    agnostic (works for left- or right-padded batches) and mirrors how numbers
    of different lengths align on columns during column-wise addition.

    Returns a (B, S) long tensor with values in ``[0, R]``.
    """
    B, S = input_ids.shape
    is_digit = (input_ids >= digit_token_start) & (input_ids < digit_token_start + num_digit_tokens)
    column_ids = torch.full((B, S), num_cyclic_positions, dtype=torch.long, device=input_ids.device)
    counter = torch.zeros(B, dtype=torch.long, device=input_ids.device)
    for t in range(S - 1, -1, -1):
        digit_t = is_digit[:, t]
        # Cyclic wrap-around: numbers wider than R columns reuse columns 0..R-1
        column_ids[:, t] = torch.where(
            digit_t,
            counter % num_cyclic_positions,
            column_ids[:, t],
        )
        counter = torch.where(digit_t, counter + 1, torch.zeros_like(counter))
    return column_ids


class Attention(nn.Module):
    """Causal multi-head self-attention (no positional encoding inside).

    Positions are already injected by the Abacus embeddings added to the token
    embeddings before the block stack, exactly as in the paper's model (standard
    dot-product attention on top of position-enriched embeddings).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")
        # Single fused linear that produces Q, K, V in one matrix multiply
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Final projection that mixes the concatenated head outputs
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

        # Reshape for multi-head attention (B, S, num_heads, head_dim) and transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # When padding tokens are present we combine causal + padding masks; otherwise
        # PyTorch's built-in causal path is used (faster and uses less memory).
        if padding_mask is not None:
            # Lower-triangular causal mask prevents attending to future tokens
            causal_mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            # Combined mask: True only where the position is both causal and non-padding
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
        # Merge heads back: (B, num_heads, S, head_dim) -> (B, S, embed_dim)
        context = context.transpose(1, 2).contiguous().view(B, S, C)
        out = self.out(context)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """A single Transformer decoder block: Pre-LN self-attention + feedforward network.

    Same layout as the RoPE baseline's blocks (Pre-LN + residual connections)
    but without rotary embeddings, since positions come from the Abacus signal.
    """

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Normalisation applied before attention (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, dropout=dropout)
        # Normalisation applied before the feedforward network (Pre-LN)
        self.norm2 = nn.LayerNorm(d_model)
        # Position-wise FFN: expands to feedforward_dim then projects back to d_model
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
        # --- Self-attention sub-layer (Pre-LN + residual) ---
        normed = self.norm1(x)
        attn_output = self.attention(normed, padding_mask=padding_mask)
        x = x + attn_output  # residual connection

        # --- Feedforward sub-layer (Pre-LN + residual) ---
        normed = self.norm2(x)
        x = x + self.feedforward(normed)  # residual connection
        return x


class AbacusTransformerModel(nn.Module):
    """Decoder-only Transformer with Abacus positional embeddings and a looped block stack.

    Architecture overview:
        token ids -> token embedding (token_embed_dim)  --\\
                    abacus column ids -> fixed sinusoidal table (abacus_dim) -> concat
                  -> hidden_size = token_embed_dim + abacus_dim
                  -> [num_layers x TransformerBlock] reused num_loops times
                  -> LayerNorm
                  -> linear projection to vocab logits

    The default ``token_embed_dim=96, abacus_dim=32`` yields the same hidden
    width (128) as a same-config RoPE baseline, keeping parameter counts
    comparable for controlled experiments.

    Config knobs:

    - ``token_embed_dim``: size of the learned token embedding.
    - ``abacus_dim``: size of the fixed positional part (must be even).
    - ``num_cyclic_positions``: number of digit-column slots ``R``.  Set it to
      the longest operand/result length seen in training; longer numbers wrap
      cyclically.  The additional +1 row is reserved for non-digit tokens.
    - ``digit_token_start`` / ``num_digit_tokens``: the contiguous id range of
      the digit tokens.  With ``CharTokenizer`` and ``charset:
      "0123456789..."`` digits start at id 5, across 10 ids.  Any charset that
      reorders digits or letters first can be handled by adjusting this.
    - ``num_layers``: blocks in the *inner* stack (this stack is reused).
    - ``num_loops``: how many times the inner stack is applied (looped /
      recurrent transformer).  Effective depth = ``num_layers * num_loops``.
    - ``max_seq_len``: accepted for config compatibility with the RoPE
      baseline but unused — there is no positional cache to precompute here.
    """

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 128,
        token_embed_dim: int = 96,
        abacus_dim: int = 32,
        num_cyclic_positions: int = 6,
        digit_token_start: int = 5,
        num_digit_tokens: int = 10,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        num_loops: int = 4,
        max_seq_len: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        super().__init__()
        if token_embed_dim + abacus_dim != hidden_size:
            raise ValueError(
                f"token_embed_dim ({token_embed_dim}) + abacus_dim ({abacus_dim}) "
                f"must equal hidden_size ({hidden_size})."
            )
        if num_loops <= 0:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")
        if num_cyclic_positions <= 0:
            raise ValueError(f"num_cyclic_positions must be positive, got {num_cyclic_positions}.")
        if num_digit_tokens <= 0 or digit_token_start < 0:
            raise ValueError("num_digit_tokens must be positive and digit_token_start non-negative.")
        self.embedding = nn.Embedding(vocab_size, token_embed_dim)
        # Abacus embedding: fixed sinusoidal table over digit-column positions
        self.abacus = AbacusPositionalEncoding(abacus_dim, num_cyclic_positions)
        self.num_cyclic_positions = num_cyclic_positions
        self.digit_token_start = digit_token_start
        self.num_digit_tokens = num_digit_tokens
        self.num_loops = num_loops
        # Stack of N decoder blocks, applied num_loops times (looped transformer)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Final linear projection: hidden_size -> vocab_size (produces raw logits)
        self.proj = nn.Linear(hidden_size, vocab_size)
        # Final layer norm applied before the projection
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S = input_ids.shape
        # Recover per-token digit-column ids from the raw token ids, then look
        # up the fixed positional embeddings.
        column_ids = compute_column_ids(
            input_ids,
            digit_token_start=self.digit_token_start,
            num_digit_tokens=self.num_digit_tokens,
            num_cyclic_positions=self.num_cyclic_positions,
        )
        abacus_embeddings = self.abacus(column_ids)

        # Scale embedding by sqrt(d_model) to keep magnitude in a reasonable range,
        # then concatenate the fixed abacus positional part.
        token_embeddings = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = torch.cat([token_embeddings, abacus_embeddings], dim=-1)

        # Build a boolean padding mask of shape (B, 1, 1, S) for the attention layers
        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # Only create a mask when there actually are padding tokens (optimisation)
            if not attention_mask.all().item():
                padding_mask = attention_mask.reshape(B, 1, 1, S).to(torch.bool)

        # Looped transformer: reuse the same inner stack num_loops times so the
        # model gains effective depth without additional parameters.  Each loop
        # iterates over the blocks in order, exactly like a recurrent pass.
        for _ in range(self.num_loops):
            for block in self.blocks:
                x = block(x, padding_mask=padding_mask)

        x = self.norm(x)
        return self.proj(x)


class CausalLMCollateFn:
    """Collate sequences for decoder-only autoregressive training.

    Pads a batch of variable-length sequences to the same length using *left*-padding
    (padding tokens are prepended rather than appended).  Left-padding keeps all real
    tokens right-aligned so a single causal mask works correctly for the whole batch.

    The trick used here is: flip each sequence, right-pad with ``pad_sequence``, then
    flip the result back -- this is equivalent to left-padding.

    ``ignore_index`` (-100 by default) is used as the label value for padding
    positions so that PyTorch's ``CrossEntropyLoss`` ignores them during training.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = pad_id if pad_id is not None else 0
        # Cross-entropy ignores positions with this label value (PyTorch convention)
        self.ignore_index = -100

    def __call__(self, batch: list[Any]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for item in batch:
            input_ids = item.train_input_ids
            labels = item.train_label_ids
            attention_mask = item.attention_mask

            # We flip the sequences to have padding on the left, which allows us to use causal masking without modification
            input_ids_list.append(input_ids.flip(0))
            labels_list.append(labels.flip(0))
            attention_mask_list.append(attention_mask.flip(0))

        # pad_sequence right-pads; flipping before and after converts that to left-padding
        padded_input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_token_id if self.pad_token_id is not None else 0,
        ).flip(1)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).flip(1)
        # Use ignore_index so the loss function skips padded label positions
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index).flip(1)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


class CausalLMCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy for logits with vocabulary in the last dimension."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return super().forward(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
