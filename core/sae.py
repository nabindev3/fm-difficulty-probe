"""TopK Sparse Autoencoder — the single shared implementation.

Both legacy repos shipped byte-identical copies of this module, so promoting it
to the core is loss-free. `expansion` and `k` are the knobs the roadmap wants
exposed (the LLM repo trained 8x SAEs, the TSFM repo 8x at d_model=512 -> 4096;
expressing both as `d_hidden = expansion * d_model` lets one config drive either).

Includes auxiliary dead-feature revival (the `aux_loss` path), which is the
reason this is "the better of each duplicated component" per Phase 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_model: int = 768, d_hidden: int | None = None,
                 expansion: int | None = None, k: int = 32, aux_k: int = 512):
        """Specify the hidden width either directly (`d_hidden`) or as a multiple
        of `d_model` (`expansion`). Exactly one of the two must be given."""
        super().__init__()
        if d_hidden is None and expansion is None:
            raise ValueError("Provide either d_hidden or expansion.")
        if d_hidden is None:
            d_hidden = expansion * d_model

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.aux_k = aux_k

        self.W_enc = nn.Parameter(torch.empty(self.d_model, self.d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))

        self.W_dec = nn.Parameter(torch.empty(self.d_hidden, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Normalize decoder columns to unit norm
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    def forward(self, x, dead_mask=None):
        # Center the input
        x_centered = x - self.b_dec

        # Encode
        pre_acts = x_centered @ self.W_enc + self.b_enc

        # Top-K routing
        top_acts, top_idx = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, top_idx, F.relu(top_acts))

        # Decode
        x_reconstruct = acts @ self.W_dec + self.b_dec

        # Aux loss for dead feature revival
        aux_loss = 0.0
        if dead_mask is not None and dead_mask.any():
            dead_pre_acts = pre_acts[:, dead_mask]
            k_aux = min(self.aux_k, dead_pre_acts.shape[-1])
            if k_aux > 0:
                aux_top_acts, aux_top_idx = torch.topk(dead_pre_acts, k_aux, dim=-1)
                aux_acts = torch.zeros_like(dead_pre_acts)
                aux_acts.scatter_(-1, aux_top_idx, F.relu(aux_top_acts))

                # Reconstruct residual using only dead features
                aux_reconstruct = aux_acts @ self.W_dec[dead_mask, :]
                residual = x - x_reconstruct.detach()
                aux_loss = F.mse_loss(aux_reconstruct, residual)

        return acts, x_reconstruct, aux_loss

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder weights normalized during training"""
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=0)

    @classmethod
    def from_checkpoint(cls, state: dict, k: int = 32, device: str = "cpu") -> "TopKSAE":
        """Rebuild a TopKSAE from a saved state dict, auto-detecting dimensions.

        Both legacy probes did this inline; centralizing it kills the
        "ran on a randomly-initialized SAE" failure mode in one place.
        """
        if "W_enc" not in state:
            raise ValueError("Not a TopKSAE checkpoint (no W_enc).")
        d_model, d_hidden = state["W_enc"].shape
        sae = cls(d_model=d_model, d_hidden=d_hidden, k=k).to(device)
        sae.load_state_dict(state)
        sae.eval()
        return sae
