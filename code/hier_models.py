#!/usr/bin/env python3
"""Hierarchical GRU model with explicit backbone / head split for FedPer-style FL."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from pvp_raw_state_models import ClassificationHead, build_encoder


class HierGRU(nn.Module):
    """Hierarchical GRU: shared backbone encoder + coarse head + per-family fine heads.

    Architecture
    ────────────
    BACKBONE  (aggregated in FL)
      input_proj  → LayerNorm → ReLU
      GRU(hidden_dim, layers, dropout)
      dropout
      → z ∈ ℝ^{hidden_dim}

    HEADS  (local in backbone-head FL)
      coarse_head: Linear→GELU→Dropout→Linear  → logits ∈ ℝ^{num_families}
      fine_heads[i]: Linear→GELU→Dropout→Linear → logits ∈ ℝ^{fine_dims[i]}
      exact_head (optional): Linear→GELU→Dropout→Linear → logits ∈ ℝ^{num_exact_classes}
    """

    # Parameter-name prefixes used to split backbone vs heads
    BACKBONE_PREFIX = "encoder."
    COARSE_HEAD_PREFIX = "coarse_head."
    FINE_HEADS_PREFIX = "fine_heads."
    EXACT_HEAD_PREFIX = "exact_head."

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_name: str = "gru",
        num_families: int,
        fine_dims: list[int],
        num_exact_classes: int = 0,
        use_exact_head: bool = False,
    ):
        super().__init__()
        self.encoder = build_encoder(
            model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.coarse_head = ClassificationHead(
            hidden_dim, num_families,
            hidden_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )
        self.fine_heads = nn.ModuleList([
            ClassificationHead(
                hidden_dim, fd,
                hidden_dim=max(128, hidden_dim // 2),
                dropout=dropout,
            )
            for fd in fine_dims
        ])
        self.use_exact_head = bool(use_exact_head) and num_exact_classes > 0
        if self.use_exact_head:
            self.exact_head = ClassificationHead(
                hidden_dim, num_exact_classes,
                hidden_dim=max(128, hidden_dim // 2),
                dropout=dropout,
            )
        else:
            self.exact_head = None
        self.hidden_dim = hidden_dim
        self.num_families = num_families
        self.fine_dims = list(fine_dims)
        self.num_exact_classes = int(num_exact_classes)

    # ── forward ──────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        z = self.encode(x, lengths)
        coarse_logits = self.coarse_head(z)
        fine_logits = [head(z) for head in self.fine_heads]
        if self.use_exact_head and self.exact_head is not None:
            exact_logits = self.exact_head(z)
            return coarse_logits, fine_logits, exact_logits
        return coarse_logits, fine_logits

    # ── backbone / head state-dict helpers ───────────────────────────────

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only encoder (backbone) parameters."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.BACKBONE_PREFIX)
        }

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only head (coarse_head + fine_heads + exact_head) parameters."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.COARSE_HEAD_PREFIX)
            or k.startswith(self.FINE_HEADS_PREFIX)
            or k.startswith(self.EXACT_HEAD_PREFIX)
        }

    def load_backbone_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load only backbone parameters, keeping heads unchanged."""
        current = self.state_dict()
        for k, v in state.items():
            if k.startswith(self.BACKBONE_PREFIX) and k in current:
                current[k] = v
        self.load_state_dict(current)

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load only head parameters, keeping backbone unchanged."""
        current = self.state_dict()
        for k, v in state.items():
            if (k.startswith(self.COARSE_HEAD_PREFIX) or
                    k.startswith(self.FINE_HEADS_PREFIX) or
                    k.startswith(self.EXACT_HEAD_PREFIX)) and k in current:
                current[k] = v
        self.load_state_dict(current)

    def full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Full state dict, detached on CPU."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def family_fine_loss_mean(
    fine_logits: list[torch.Tensor],
    y_coarse: torch.Tensor,
    y_fine: torch.Tensor,
    fine_class_weights: list[torch.Tensor | None],
) -> torch.Tensor:
    """Compute mean cross-entropy over fine predictions, masked per family."""
    total_ce = torch.tensor(0.0, device=y_coarse.device)
    total_n = 0
    for fam_id, logits in enumerate(fine_logits):
        mask = y_coarse == fam_id
        if not bool(mask.any()):
            continue
        w = fine_class_weights[fam_id] if fam_id < len(fine_class_weights) else None
        ce = F.cross_entropy(logits[mask], y_fine[mask], weight=w, reduction="sum")
        total_ce = total_ce + ce
        total_n += int(mask.sum().item())
    if total_n <= 0:
        return torch.tensor(0.0, device=y_coarse.device)
    return total_ce / float(total_n)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_state_dicts(
    weighted_states: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts by sample count."""
    total_weight = sum(max(1, n) for _, n in weighted_states)
    agg: dict[str, torch.Tensor] = {}
    for state, n in weighted_states:
        w = max(1, n) / max(1, total_weight)
        for k, v in state.items():
            if k not in agg:
                agg[k] = v.clone().float() * w
            else:
                agg[k] = agg[k] + v.float() * w
    return agg


def model_config_dict(
    input_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    model_name: str,
    num_families: int,
    fine_dims: list[int],
) -> dict:
    """Build the kwargs dict for HierGRU constructor."""
    return dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        model_name=model_name,
        num_families=num_families,
        fine_dims=fine_dims,
    )


# ---------------------------------------------------------------------------
# Flat coarse-only model
# ---------------------------------------------------------------------------

class FlatCoarseOnlyGRU(nn.Module):
    """Shared GRU encoder + single coarse classifier — no fine heads.

    Use for taxonomy ablation: measure honest coarse accuracy in isolation.
    """

    BACKBONE_PREFIX = "encoder."
    HEAD_PREFIX = "coarse_head."

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_name: str = "gru",
        num_families: int,
    ):
        super().__init__()
        self.encoder = build_encoder(
            model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.coarse_head = ClassificationHead(
            hidden_dim, num_families,
            hidden_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.num_families = num_families

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, lengths)
        return self.coarse_head(z)

    def full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.BACKBONE_PREFIX)
        }


# ---------------------------------------------------------------------------
# Multitask direct-exact model
# ---------------------------------------------------------------------------

class MultitaskDirectExactGRU(nn.Module):
    """Shared GRU encoder + coarse head (auxiliary) + direct exact head.

    The coarse prediction is auxiliary — the exact action is predicted directly
    without any routing through coarse families.
    """

    BACKBONE_PREFIX = "encoder."
    COARSE_HEAD_PREFIX = "coarse_head."
    EXACT_HEAD_PREFIX = "exact_head."

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_name: str = "gru",
        num_families: int,
        num_exact_classes: int,
    ):
        super().__init__()
        self.encoder = build_encoder(
            model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.coarse_head = ClassificationHead(
            hidden_dim, num_families,
            hidden_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )
        self.exact_head = ClassificationHead(
            hidden_dim, num_exact_classes,
            hidden_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.num_families = num_families
        self.num_exact_classes = num_exact_classes

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (coarse_logits, exact_logits)."""
        z = self.encode(x, lengths)
        coarse_logits = self.coarse_head(z)
        exact_logits = self.exact_head(z)
        return coarse_logits, exact_logits

    def full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.BACKBONE_PREFIX)
        }


# ---------------------------------------------------------------------------
# Soft-router exact model
# ---------------------------------------------------------------------------

class SoftRouterExactGRU(nn.Module):
    """Coarse head + per-family fine heads with soft top-k routing to exact logits.

    Instead of hard argmax on the coarse prediction, this model forms exact
    logits as a weighted mixture of the top-k families' fine-head outputs,
    weighted by their coarse softmax probabilities.

    When the top-1 coarse probability is 1.0, this reduces exactly to hard
    routing (i.e. equivalent to HierGRU).
    """

    BACKBONE_PREFIX = "encoder."
    COARSE_HEAD_PREFIX = "coarse_head."
    FINE_HEADS_PREFIX = "fine_heads."

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_name: str = "gru",
        num_families: int,
        fine_dims: list[int],
        num_exact_classes: int,
        family_fine_to_exact_ids: dict[int, dict[int, int]],
        top_k: int = 2,
        coarse_temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = build_encoder(
            model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.coarse_head = ClassificationHead(
            hidden_dim, num_families,
            hidden_dim=max(128, hidden_dim // 2),
            dropout=dropout,
        )
        self.fine_heads = nn.ModuleList([
            ClassificationHead(
                hidden_dim, fd,
                hidden_dim=max(128, hidden_dim // 2),
                dropout=dropout,
            )
            for fd in fine_dims
        ])
        self.hidden_dim = hidden_dim
        self.num_families = num_families
        self.fine_dims = list(fine_dims)
        self.num_exact_classes = num_exact_classes
        self.top_k = min(top_k, num_families)
        self.coarse_temperature = coarse_temperature

        # Build scatter-index buffers: for each family, a mapping from fine_id → exact_id
        # Stored as registered buffers for device placement.
        for fam_id in range(num_families):
            fam_map = family_fine_to_exact_ids.get(fam_id, {})
            idx = torch.zeros(fine_dims[fam_id], dtype=torch.long)
            for fine_id, exact_id in fam_map.items():
                if 0 <= fine_id < fine_dims[fam_id] and 0 <= exact_id < num_exact_classes:
                    idx[fine_id] = exact_id
            self.register_buffer(f"_fine_to_exact_{fam_id}", idx)

    def _fine_to_exact_index(self, fam_id: int) -> torch.Tensor:
        return getattr(self, f"_fine_to_exact_{fam_id}")

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Returns (coarse_logits, fine_logits_list, soft_exact_logits)."""
        z = self.encode(x, lengths)
        coarse_logits = self.coarse_head(z)
        fine_logits = [head(z) for head in self.fine_heads]

        # Soft mixture of fine-head outputs weighted by coarse probabilities
        soft_exact = self._compute_soft_exact(coarse_logits, fine_logits)
        return coarse_logits, fine_logits, soft_exact

    def _compute_soft_exact(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        """Form exact-action logits via weighted top-k family mixture."""
        B = coarse_logits.shape[0]
        device = coarse_logits.device

        # Apply temperature scaling
        scaled_logits = coarse_logits / max(1e-4, self.coarse_temperature)
        coarse_probs = F.softmax(scaled_logits, dim=1)  # (B, num_families)
        topk = torch.topk(coarse_probs, k=self.top_k, dim=1)  # values, indices
        topk_probs = topk.values   # (B, top_k)
        topk_fams = topk.indices   # (B, top_k)

        # For top_k=1, enforce exact equivalence to hard routing.
        if self.top_k == 1:
            exact_logits = torch.full((B, self.num_exact_classes), -1e9, device=device)
            fam_ids = topk_fams[:, 0]
            for fam_id in fam_ids.unique().tolist():
                fam_mask = fam_ids == fam_id
                if not fam_mask.any():
                    continue

                fam_fine_logits = fine_logits[fam_id][fam_mask]  # (n, fine_dim)
                scatter_idx = self._fine_to_exact_index(fam_id)  # (fine_dim,)
                expanded = torch.full(
                    (fam_fine_logits.shape[0], self.num_exact_classes),
                    -1e9,
                    device=device,
                )
                expanded.scatter_(
                    1,
                    scatter_idx.unsqueeze(0).expand(fam_fine_logits.shape[0], -1),
                    fam_fine_logits,
                )
                exact_logits[fam_mask] = expanded
            return exact_logits

        # Renormalize top-k probabilities to sum to 1
        topk_probs = topk_probs / (topk_probs.sum(dim=1, keepdim=True) + 1e-12)

        exact_logits = torch.zeros(B, self.num_exact_classes, device=device)

        for k_idx in range(self.top_k):
            fam_ids = topk_fams[:, k_idx]   # (B,)
            weights = topk_probs[:, k_idx]  # (B,)

            for fam_id in fam_ids.unique().tolist():
                fam_mask = fam_ids == fam_id
                if not fam_mask.any():
                    continue

                fam_fine_logits = fine_logits[fam_id][fam_mask]  # (n, fine_dim)
                scatter_idx = self._fine_to_exact_index(fam_id)  # (fine_dim,)

                # Scatter fine logits into exact-action space
                expanded = torch.zeros(fam_fine_logits.shape[0], self.num_exact_classes, device=device)
                expanded.scatter_(1, scatter_idx.unsqueeze(0).expand(fam_fine_logits.shape[0], -1), fam_fine_logits)

                # Weight by coarse probability
                w = weights[fam_mask].unsqueeze(1)  # (n, 1)
                exact_logits[fam_mask] = exact_logits[fam_mask] + expanded * w

        return exact_logits

    def full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.BACKBONE_PREFIX)
        }

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.COARSE_HEAD_PREFIX)
            or k.startswith(self.FINE_HEADS_PREFIX)
        }


# ---------------------------------------------------------------------------
# Race-specific heads model
# ---------------------------------------------------------------------------

class HierGRURaceHeads(nn.Module):
    """Hierarchical GRU: shared backbone + race-specific coarse+fine heads.

    Architecture
    ────────────
    BACKBONE  (aggregated in FL)
      input_proj  → LayerNorm → ReLU
      GRU(hidden_dim, layers, dropout)
      dropout
      → z ∈ ℝ^{hidden_dim}

    RACE-SPECIFIC HEADS  (local in backbone-head FL)
      For each race {Protoss, Terran, Zerg}:
        coarse_head: Linear→ReLU→Dropout→Linear  → logits ∈ ℝ^{num_coarse}
        fine_heads[i]: Linear→ReLU→Dropout→Linear → logits ∈ ℝ^{fine_dims[i]}

    Routing: Each sample routes to the correct race head based on its race label.
    """

    # Parameter-name prefixes used to split backbone vs heads
    BACKBONE_PREFIX = "encoder."
    RACE_HEADS_PREFIX = "race_heads."

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        model_name: str = "gru",
        race_fine_dims: dict[str, list[int]],  # {race → [fine_dims for each coarse class]}
        race_num_coarse: dict[str, int],  # {race → num coarse classes}
        races: list[str] = None,  # ["Prot", "Terr", "Zerg"]
        num_exact_classes: int = 0,
    ):
        super().__init__()
        if races is None:
            races = ["Prot", "Terr", "Zerg"]
        self.races = list(races)
        self.race_num_coarse = {str(r): int(nc) for r, nc in (race_num_coarse or {}).items()}
        self.race_fine_dims = {str(r): list(fd) for r, fd in (race_fine_dims or {}).items()}
        self.num_exact_classes = int(num_exact_classes)
        self.use_exact_head = self.num_exact_classes > 0

        self.encoder = build_encoder(
            model_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )

        # Build race-specific heads
        self.race_heads = nn.ModuleDict()
        for race in self.races:
            num_coarse = self.race_num_coarse.get(race, 1)
            fine_dims = self.race_fine_dims.get(race, [1] * num_coarse)

            head_dict = nn.ModuleDict()
            head_dict["coarse"] = ClassificationHead(
                hidden_dim, num_coarse,
                hidden_dim=max(128, hidden_dim // 2),
                dropout=dropout,
            )
            fine_heads = nn.ModuleList([
                ClassificationHead(
                    hidden_dim, fd,
                    hidden_dim=max(128, hidden_dim // 2),
                    dropout=dropout,
                )
                for fd in fine_dims
            ])
            head_dict["fine_heads"] = fine_heads
            # Per-race auxiliary exact head (direct z → all exact actions)
            if self.use_exact_head:
                head_dict["exact"] = ClassificationHead(
                    hidden_dim, self.num_exact_classes,
                    hidden_dim=max(128, hidden_dim // 2),
                    dropout=dropout,
                )
            self.race_heads[race] = head_dict

        self.hidden_dim = hidden_dim

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lengths)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        race_ids: torch.Tensor = None,  # (B,) with values 0, 1, 2 for Prot, Terr, Zerg
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Returns a dict with per-race outputs:
        {
            "Prot": {"coarse": ..., "fine": [...], "exact": ... (if enabled)},
            "Terr": {...},
            "Zerg": {...},
        }
        If race_ids is None, returns all races' outputs for all samples.
        """
        z = self.encode(x, lengths)

        if race_ids is None:
            # Return all races' outputs for all samples
            result = {}
            for race in self.races:
                head = self.race_heads[race]
                coarse_logits = head["coarse"](z)
                fine_logits = [h(z) for h in head["fine_heads"]]
                entry = {
                    "coarse": coarse_logits,
                    "fine": fine_logits,
                }
                if self.use_exact_head and "exact" in head:
                    entry["exact"] = head["exact"](z)
                result[race] = entry
            return result
        else:
            # Route samples to their respective race heads
            result = {}
            for race in self.races:
                mask = race_ids == self.races.index(race)
                if not mask.any():
                    continue
                z_masked = z[mask]
                head = self.race_heads[race]
                coarse_logits = head["coarse"](z_masked)
                fine_logits = [h(z_masked) for h in head["fine_heads"]]
                entry = {
                    "coarse": coarse_logits,
                    "fine": fine_logits,
                    "mask": mask,
                }
                if self.use_exact_head and "exact" in head:
                    entry["exact"] = head["exact"](z_masked)
                result[race] = entry
            return result

    def backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only encoder (backbone) parameters."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.BACKBONE_PREFIX)
        }

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only race-specific head parameters."""
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith(self.RACE_HEADS_PREFIX)
        }

    def load_backbone_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load only backbone parameters, keeping heads unchanged."""
        current = self.state_dict()
        for k, v in state.items():
            if k.startswith(self.BACKBONE_PREFIX) and k in current:
                current[k] = v
        self.load_state_dict(current)

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load only head parameters, keeping backbone unchanged."""
        current = self.state_dict()
        for k, v in state.items():
            if k.startswith(self.RACE_HEADS_PREFIX) and k in current:
                current[k] = v
        self.load_state_dict(current)

    def full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Full state dict, detached on CPU."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

ARCH_NAMES = [
    "hard_hierarchical",
    "flat_coarse_only",
    "multitask_direct_exact",
    "soft_router_exact",
    "race_heads",
]


def build_model(arch: str, **kwargs) -> nn.Module:
    """Factory: build a model by architecture name.

    Supported architectures: hard_hierarchical, flat_coarse_only,
    multitask_direct_exact, soft_router_exact, race_heads.
    """
    arch = str(arch).strip().lower()
    if arch == "hard_hierarchical":
        return HierGRU(**kwargs)
    if arch == "flat_coarse_only":
        return FlatCoarseOnlyGRU(**kwargs)
    if arch == "multitask_direct_exact":
        return MultitaskDirectExactGRU(**kwargs)
    if arch == "soft_router_exact":
        return SoftRouterExactGRU(**kwargs)
    if arch == "race_heads":
        return HierGRURaceHeads(**kwargs)
    raise ValueError(f"Unknown architecture: {arch}. Supported: {ARCH_NAMES}")

