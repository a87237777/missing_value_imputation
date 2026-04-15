# ==== GAIN-style imputer (continuous features) ====
"""
gan_impute.py
-------------
Standalone lightweight GAIN-style (GAN) imputation for tabular data.

Usage:
    from gan_impute import impute_gain
    X_hat = impute_gain(X_miss, random_state=0, epochs=100)

Notes:
    * Observed entries are preserved exactly in the output.
    * Features are standardized using observed entries only.
    * Requires PyTorch.
"""
from __future__ import annotations
import numpy as np
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    torch = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _standardize_observed(X):
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sd_safe = np.where((np.isfinite(sd)) & (sd >= 1e-12), sd, 1.0)
    Xs = (X - mu_safe) / sd_safe
    Xs = np.where(np.isnan(Xs), 0.0, Xs)
    return Xs, mu_safe, sd_safe


def impute_gain(
    X_miss,
    random_state: int | None = 0,
    epochs: int = 100,
    g_hidden: int | None = None,
    d_hidden: int | None = None,
    lr_g: float = 1e-3,
    lr_d: float = 1e-3,
    adv_weight: float = 1.0,
    rec_weight: float = 1.0,
    verbose: bool = False,
):

    if torch is None:
        raise ImportError(f"PyTorch not available: {_IMPORT_ERR}")

    X_miss = np.asarray(X_miss, float)
    n, p = X_miss.shape
    M = ~np.isnan(X_miss)
    M_float = M.astype(float)

    Xs, mu, sd = _standardize_observed(X_miss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if random_state is not None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

    X_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_float, dtype=torch.float32, device=device)

    gh = int(g_hidden) if g_hidden is not None else max(32, p)
    dh = int(d_hidden) if d_hidden is not None else max(32, p)

    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*p, gh), nn.ReLU(),
                nn.Linear(gh, gh), nn.ReLU(),
                nn.Linear(gh, p), nn.Tanh(),   # outputs ~[-1,1]
            )
        def forward(self, x):
            return self.net(x)

    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*p, dh), nn.ReLU(),
                nn.Linear(dh, dh), nn.ReLU(),
                nn.Linear(dh, p), nn.Sigmoid(),  # per-feature prob observed
            )
        def forward(self, x):
            return self.net(x)

    G = Gen().to(device)
    D = Disc().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCELoss(reduction='none')

    def _mask_avg(elems, mask):
        return (elems * mask).sum() / (mask.sum() + 1e-8)

    for ep in range(int(epochs)):
        Z = torch.randn_like(X_t)
        X_tilde = M_t * X_t + (1 - M_t) * Z
        G_in = torch.cat([X_tilde, M_t], dim=1)

        with torch.no_grad():
            G_out = G(G_in)
            X_hat = M_t * X_t + (1 - M_t) * G_out

        D_in_real = torch.cat([X_t, M_t], dim=1)
        D_in_fake = torch.cat([X_hat, M_t], dim=1)
        D_real = D(D_in_real)
        D_fake = D(D_in_fake)

        D_loss_real = _mask_avg(bce(D_real, torch.ones_like(M_t)), M_t)
        D_loss_fake = _mask_avg(bce(D_fake, torch.zeros_like(M_t)), (1 - M_t))
        D_loss = D_loss_real + D_loss_fake

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        Z = torch.randn_like(X_t)
        X_tilde = M_t * X_t + (1 - M_t) * Z
        G_in = torch.cat([X_tilde, M_t], dim=1)
        G_out = G(G_in)
        X_hat = M_t * X_t + (1 - M_t) * G_out

        D_hat = D(torch.cat([X_hat, M_t], dim=1))

        G_adv = _mask_avg(bce(D_hat, torch.ones_like(M_t)), (1 - M_t))
        G_rec = _mask_avg((X_hat - X_t)**2, M_t)

        G_loss = rec_weight * G_rec + adv_weight * G_adv

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[GAIN] epoch {ep+1}/{epochs} D={D_loss.item():.4f} Grec={G_rec.item():.4f} Gadv={G_adv.item():.4f}")

    # Final imputation
    with torch.no_grad():
        Z = torch.randn_like(X_t)
        X_tilde = M_t * X_t + (1 - M_t) * Z
        G_in = torch.cat([X_tilde, M_t], dim=1)
        G_out = G(G_in)
        X_imp_std = M_t * X_t + (1 - M_t) * G_out
        X_imp_std = X_imp_std.cpu().numpy()

    X_imp = X_imp_std * sd + mu
    X_out = np.array(X_imp, float)
    X_out[M] = X_miss[M]

    return X_out
