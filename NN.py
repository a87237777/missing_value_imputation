# NN.py
"""
SimpleNN imputation (masked autoencoder) for tabular data with NaNs.
Usage:
    from simple_nn_impute import impute_nn
    X_hat = impute_nn(X_miss, random_state=0, epochs=50)

Notes:
  * Observed entries are reconstructed and used in the loss; missing entries are ignored by the loss.
  * Output restores observed cells exactly to original values.
  * Features are standardized using observed entries only.
  * Requires PyTorch.
"""
from __future__ import annotations
import numpy as np
import warnings


import torch
import torch.nn as nn
import torch.optim as optim


def _standardize_observed(X: np.ndarray):
    """Standardize per column using observed entries only; missing -> 0 after scaling."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sd_safe = np.where((np.isfinite(sd)) & (sd >= 1e-12), sd, 1.0)
    Xs = (X - mu_safe) / sd_safe
    Xs = np.where(np.isnan(Xs), 0.0, Xs)
    return Xs, mu_safe, sd_safe


class _MaskedMSE(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff2 = (pred - target) ** 2
        num_obs = mask.sum().clamp_min(1.0)
        return (diff2 * mask).sum() / num_obs


def impute_nn(
    X_miss,
    *,
    random_state: int | None = 0,
    epochs: int = 50,
    hidden: tuple[int, ...] = (128, 64),
    dropout: float = 0.0,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    early_stop_patience: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    """
    Deterministic masked autoencoder imputation.

    Parameters
    ----------
    X_miss : array-like (n, p)
        Matrix with NaNs for missing entries.
    random_state : int, optional
    epochs : int, default=50
    hidden : tuple[int,...], default=(128, 64)
        Hidden layer sizes for the encoder; decoder mirrors back to p.
    dropout : float, default=0.0
    lr : float, default=1e-3
    batch_size : int, default=256
    weight_decay : float, default=0.0
    early_stop_patience : int, default=10
    verbose : bool, default=False

    Returns
    -------
    X_hat : np.ndarray (n, p)
        Imputed matrix; observed entries restored exactly.
    """
    if torch is None:
        raise ImportError(f"PyTorch not available: {_IMPORT_ERR}")

    X_miss = np.asarray(X_miss, float)
    n, p = X_miss.shape
    obs_mask = ~np.isnan(X_miss)
    obs_mask_f = obs_mask.astype(np.float32)

    Xs, mu, sd = _standardize_observed(X_miss)

    # torch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if random_state is not None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

    X_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    M_t = torch.tensor(obs_mask_f, dtype=torch.float32, device=device)


    layers_e = []
    in_dim = p
    for h in hidden:
        layers_e += [nn.Linear(in_dim, h), nn.ReLU()]
        if dropout > 0:
            layers_e += [nn.Dropout(dropout)]
        in_dim = h
    encoder = nn.Sequential(*layers_e)

    layers_d = []
    dec_dims = list(hidden[::-1]) + [p]
    in_dim = hidden[-1] if hidden else p
    for j, h in enumerate(dec_dims):
        out_is_final = (j == len(dec_dims) - 1)
        layers_d += [nn.Linear(in_dim, h)]
        if not out_is_final:
            layers_d += [nn.ReLU()]
            if dropout > 0:
                layers_d += [nn.Dropout(dropout)]
        in_dim = h
    decoder = nn.Sequential(*layers_d)

    class AutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    net = AutoEncoder().to(device)
    criterion = _MaskedMSE()
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    idx = np.arange(n)
    val_size = max(1, int(0.1 * n))
    if val_size >= n:  # degenerate tiny n
        val_size = 1
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, M_tr = X_t[tr_idx], M_t[tr_idx]
    X_val, M_val = X_t[val_idx], M_t[val_idx]

    def batches(Xb, Mb, bs):
        m = Xb.shape[0]
        for s in range(0, m, bs):
            e = min(m, s + bs)
            yield Xb[s:e], Mb[s:e]

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(int(epochs)):
        net.train()
        tr_loss = 0.0
        for xb, mb in batches(X_tr, M_tr, batch_size):
            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            loss = criterion(pred, xb, mb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, X_tr.size(0))

        # val
        net.eval()
        with torch.no_grad():
            pred_val = net(X_val)
            val_loss = criterion(pred_val, X_val, M_val).item()

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[SimpleNN] epoch {ep+1}/{epochs} train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(early_stop_patience):
                if verbose:
                    print(f"[SimpleNN] early stop at epoch {ep+1} (best val={best_val:.4f})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        X_pred = net(X_t).cpu().numpy()

    X_hat = X_pred * sd + mu

    X_out = np.array(X_hat, dtype=float)
    X_out[obs_mask] = X_miss[obs_mask]
    return X_out
