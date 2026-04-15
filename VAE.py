# ! pip install torch
"""
VAE.py
-------------
Standalone VAE-based imputation for tabular data with missing values (np.nan).

Usage:
    from vae_impute import impute_vae
    X_hat = impute_vae(X_miss, random_state=0, epochs=50)

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
    """Standardize per column using observed entries only; nan -> 0 after scaling."""
    X = np.asarray(X, float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    sd_safe = np.where((np.isfinite(sd)) & (sd >= 1e-12), sd, 1.0)
    Xs = (X - mu_safe) / sd_safe
    Xs = np.where(np.isnan(Xs), 0.0, Xs)
    return Xs, mu_safe, sd_safe


def impute_vae(
    X_miss,
    random_state: int | None = 0,
    epochs: int = 50,
    hidden: int | None = None,
    latent: int | None = None,
    lr: float = 1e-3,
    kl_weight: float = 1e-3,
    verbose: bool = False,
):
    """
    VAE imputation. Returns X_hat with observed entries restored to original values.
    """
    if torch is None:
        raise ImportError(f"PyTorch not available: {_IMPORT_ERR}")

    X_miss = np.asarray(X_miss, float)
    n, p = X_miss.shape
    M = ~np.isnan(X_miss)   # observed mask
    M_float = M.astype(float)

    Xs, mu, sd = _standardize_observed(X_miss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if random_state is not None:
        torch.manual_seed(int(random_state))
        np.random.seed(int(random_state))

    X_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_float, dtype=torch.float32, device=device)

    h = int(hidden) if hidden is not None else max(32, p)
    zdim = int(latent) if latent is not None else max(2, min(8, p // 2))

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Linear(p, h)
            self.enc2_mu = nn.Linear(h, zdim)
            self.enc2_logvar = nn.Linear(h, zdim)
            self.dec1 = nn.Linear(zdim, h)
            self.dec2 = nn.Linear(h, p)
            self.act = nn.ReLU()

        def encode(self, x):
            h1 = self.act(self.enc1(x))
            return self.enc2_mu(h1), self.enc2_logvar(h1)

        def reparam(self, mu, logvar):
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps

        def decode(self, z):
            h1 = self.act(self.dec1(z))
            return self.dec2(h1)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparam(mu, logvar)
            xhat = self.decode(z)
            return xhat, mu, logvar

    vae = VAE().to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='none')

    for ep in range(int(epochs)):
        vae.train()
        opt.zero_grad()
        xhat, mu_t, logvar_t = vae(X_t)

        # Masked reconstruction over observed entries only
        recon_e = mse(xhat, X_t) * M_t
        denom = M_t.sum()
        recon = recon_e.sum() / (denom + 1e-8)

        # KL
        kl = -0.5 * torch.sum(1 + logvar_t - mu_t.pow(2) - logvar_t.exp()) / (n + 1e-8)
        loss = recon + kl_weight * kl
        loss.backward()
        opt.step()

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"[VAE] epoch {ep+1}/{epochs} loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")

    vae.eval()
    with torch.no_grad():
        mu_t, logvar_t = vae.encode(X_t)
        z = mu_t  # mean of posterior for stability
        xhat = vae.decode(z).cpu().numpy()

    X_vae = xhat * sd + mu
    # Restore observed entries exactly
    X_out = np.array(X_vae, float)
    X_out[M] = X_miss[M]

    return X_out
