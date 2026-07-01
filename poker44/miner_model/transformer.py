"""Hierarchical Transformer for Poker44 bot detection.

Architecture: Hand Encoder (action tokens → hand vector) +
              Chunk Encoder (hand vectors → session vector) + MLP.

Lives in the poker44 package so joblib can unpickle from any working directory.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------

_ACT_TYPES = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4}
_ACT_PAD = 5

_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_STREET_PAD = 4

_BB_BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)
_AMOUNT_NONE = 0   # fold/check/call with no amount
_AMOUNT_PAD = 16   # padding index (out of 0..16)

_HERO_PAD = 2  # 0=other, 1=hero, 2=pad

MAX_ACTIONS = 20   # pad hand sequences to this length
MAX_HANDS   = 64   # pad session sequences to this length


def _amount_bucket(norm_bb: float) -> int:
    """Map a normalized BB amount to a bucket index (1..15); 0 if no amount."""
    if norm_bb <= 0:
        return _AMOUNT_NONE
    return 1 + min(range(len(_BB_BUCKETS)), key=lambda i: abs(_BB_BUCKETS[i] - norm_bb))


def tokenize_hand(hand: Dict[str, Any]) -> np.ndarray:
    """Convert one hand dict → (MAX_ACTIONS, 4) int array.

    Columns: [action_type_id, street_id, amount_bucket_id, is_hero_id]
    Padding value: PAD index for each column.
    """
    actions = hand.get("actions") or []
    meta = hand.get("metadata") or {}
    try:
        hero_seat = int(meta.get("hero_seat", 0) or 0)
    except (TypeError, ValueError):
        hero_seat = 0

    tokens = np.full((MAX_ACTIONS, 4), fill_value=0, dtype=np.int64)
    # Set padding defaults: use PAD indices for empty slots
    tokens[:, 0] = _ACT_PAD
    tokens[:, 1] = _STREET_PAD
    tokens[:, 2] = _AMOUNT_PAD
    tokens[:, 3] = _HERO_PAD

    for i, a in enumerate(actions[:MAX_ACTIONS]):
        if not isinstance(a, dict):
            continue
        at = str(a.get("action_type", "") or "").lower()
        st = str(a.get("street", "") or "").lower()
        norm_bb = float(a.get("normalized_amount_bb", 0.0) or 0.0)
        try:
            seat = int(a.get("actor_seat", 0) or 0)
        except (TypeError, ValueError):
            seat = 0

        tokens[i, 0] = _ACT_TYPES.get(at, _ACT_PAD)
        tokens[i, 1] = _STREETS.get(st, _STREET_PAD)
        tokens[i, 2] = _amount_bucket(norm_bb)
        tokens[i, 3] = 1 if (seat != 0 and seat == hero_seat) else 0

    return tokens


def tokenize_session(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """Convert one session (list of hand dicts) → (MAX_HANDS, MAX_ACTIONS, 4) int array."""
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    out = np.full((MAX_HANDS, MAX_ACTIONS, 4), fill_value=0, dtype=np.int64)
    # Fill with padding
    out[:, :, 0] = _ACT_PAD
    out[:, :, 1] = _STREET_PAD
    out[:, :, 2] = _AMOUNT_PAD
    out[:, :, 3] = _HERO_PAD

    for hi, hand in enumerate(hands[:MAX_HANDS]):
        out[hi] = tokenize_hand(hand)

    return out


# ---------------------------------------------------------------------------
# PyTorch model (only available when torch is installed)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _HandEncoder(nn.Module):
        """Encode one hand (sequence of actions) → fixed-size vector."""

        def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            d_per = d_model // 4  # 16 each for 4 fields; total = d_model
            self.act_emb    = nn.Embedding(6,  d_per, padding_idx=_ACT_PAD)
            self.street_emb = nn.Embedding(5,  d_per, padding_idx=_STREET_PAD)
            self.amount_emb = nn.Embedding(17, d_per, padding_idx=_AMOUNT_PAD)
            self.hero_emb   = nn.Embedding(3,  d_per, padding_idx=_HERO_PAD)
            enc_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers, enable_nested_tensor=False)
            self.attn_q = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            """
            x: (B, T, 4) int64
            padding_mask: (B, T) bool — True where padded
            Returns: (B, d_model)
            """
            h = torch.cat([
                self.act_emb(x[..., 0]),
                self.street_emb(x[..., 1]),
                self.amount_emb(x[..., 2]),
                self.hero_emb(x[..., 3]),
            ], dim=-1)  # (B, T, d_model)
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            # Attention-pool over non-pad positions
            w = self.attn_q(h).squeeze(-1)  # (B, T)
            w = w.masked_fill(padding_mask, -1e9)
            w = torch.softmax(w, dim=-1).unsqueeze(-1)  # (B, T, 1)
            return (w * h).sum(dim=1)  # (B, d_model)

    class _ChunkEncoder(nn.Module):
        """Encode a sequence of hand vectors → single session vector."""

        def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.2):
            super().__init__()
            enc_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, 1, enable_nested_tensor=False)
            self.attn_q = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            """
            x: (1, N_hands, d_model)
            padding_mask: (1, N_hands) bool
            Returns: (1, d_model)
            """
            h = self.encoder(x, src_key_padding_mask=padding_mask)
            w = self.attn_q(h).squeeze(-1)
            w = w.masked_fill(padding_mask, -1e9)
            w = torch.softmax(w, dim=-1).unsqueeze(-1)
            return (w * h).sum(dim=1)

    class ChunkTransformer(nn.Module):
        """Hierarchical Transformer: Hand → Session → Bot probability."""

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            n_hand_layers: int = 2,
            dropout: float = 0.25,
        ):
            super().__init__()
            self.d_model = d_model
            self.hand_encoder = _HandEncoder(d_model, n_heads, n_hand_layers, dropout)
            self.chunk_encoder = _ChunkEncoder(d_model, n_heads, dropout)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def encode_session(self, session_tensor: torch.Tensor) -> torch.Tensor:
            """session_tensor: (MAX_HANDS, MAX_ACTIONS, 4) int64 on the right device."""
            # Find real hands: at least one action position is not ACT_PAD
            act_pad_mask = (session_tensor[:, :, 0] == _ACT_PAD)  # (MAX_HANDS, MAX_ACTIONS)
            hand_is_real = ~act_pad_mask.all(dim=-1)  # (MAX_HANDS,) True for non-empty hands

            if not hand_is_real.any():
                # Empty session — return zero vector
                return torch.zeros(1, self.d_model, dtype=torch.float32)

            # Only encode real hands to avoid NaN from all-padded attention
            real_hands = session_tensor[hand_is_real]           # (n_real, MAX_ACTIONS, 4)
            real_act_pad = act_pad_mask[hand_is_real]           # (n_real, MAX_ACTIONS)

            hand_vecs = self.hand_encoder(real_hands, real_act_pad)  # (n_real, d_model)

            n_real = hand_vecs.shape[0]
            hand_input = hand_vecs.unsqueeze(0)                       # (1, n_real, d_model)
            hand_pad_mask = torch.zeros(1, n_real, dtype=torch.bool)  # no padding

            chunk_vec = self.chunk_encoder(hand_input, hand_pad_mask)  # (1, d_model)
            return chunk_vec  # (1, d_model)

        def forward(self, session_tensor: torch.Tensor) -> torch.Tensor:
            """Returns scalar bot probability."""
            vec = self.encode_session(session_tensor)  # (1, d_model)
            logit = self.head(vec)  # (1, 1)
            return torch.sigmoid(logit).squeeze()  # scalar


class TransformerScorer:
    """Sklearn-compatible wrapper around ChunkTransformer.

    fit(sessions, y) trains from list of raw chunk lists.
    predict_proba(sessions) returns (N, 2) ndarray.
    predict_proba_from_tensors(tensors) takes pre-tokenized tensors.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_hand_layers: int = 2,
        dropout: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 16,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_hand_layers = n_hand_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional["ChunkTransformer"] = None

    def _to_tensors(self, sessions: List[List[Dict]]) -> List[torch.Tensor]:
        """Tokenize a list of sessions → list of (MAX_HANDS, MAX_ACTIONS, 4) tensors."""
        result = []
        for chunk in sessions:
            arr = tokenize_session(chunk)
            result.append(torch.from_numpy(arr))
        return result

    def fit(self, sessions: List[List[Dict]], y: np.ndarray) -> "TransformerScorer":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")
        self.model = ChunkTransformer(
            self.d_model, self.n_heads, self.n_hand_layers, self.dropout
        )
        tensors = self._to_tensors(sessions)
        labels = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr / 20
        )

        n = len(tensors)
        idx = list(range(n))
        self.model.train()
        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0)
                for i in batch_idx:
                    prob = self.model(tensors[i])
                    loss = nn.functional.binary_cross_entropy(
                        prob.unsqueeze(0), labels[i].unsqueeze(0)
                    )
                    batch_loss = batch_loss + loss
                (batch_loss / len(batch_idx)).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / n
                print(f"[transformer] epoch {epoch+1}/{self.epochs} loss={avg:.4f}", flush=True)
        return self

    def predict_proba(self, sessions: List[List[Dict]]) -> np.ndarray:
        """Returns (N, 2) ndarray of [human_prob, bot_prob]."""
        if self.model is None:
            raise RuntimeError("TransformerScorer not fitted")
        tensors = self._to_tensors(sessions)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for t in tensors:
                p = float(self.model(t).item())
                probs.append(p)
        arr = np.array(probs)
        return np.column_stack([1 - arr, arr])
