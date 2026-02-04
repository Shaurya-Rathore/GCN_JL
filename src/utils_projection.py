# src/utils_projection.py
import os, json
import numpy as np
import torch

def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def _to_torch(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)
    return x

def _jl_matrix(d_in, d_out, seed=0, orth=False):
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    R = torch.randn(d_in, d_out, generator=g) / (d_out ** 0.5)
    if orth:
        Q, _ = torch.linalg.qr(R, mode='reduced')
        R = Q[:, :d_out]
    return R  # CPU tensor

def _pca_fit(X_np, d_out, whiten=False):
    mu = X_np.mean(axis=0, keepdims=True)
    Xc = X_np - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :d_out]            # [d, d_out]
    S = S[:d_out]
    meta = {'mu': mu.squeeze(0), 'S': S, 'V': V, 'whiten': bool(whiten)}
    return meta

def _pca_transform(X_np, meta):
    Xc = X_np - meta['mu']
    Z = Xc @ meta['V']             # [N, d_out]
    if meta['whiten']:
        Z = Z / (meta['S'] + 1e-8)
    return Z

def _tsne_fit_transform(X_np, d_out, perplexity=30.0, lr=200.0, seed=0):
    # Intended for one-time DR; non-parametric (cannot transform new points)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=d_out, perplexity=perplexity, learning_rate=lr,
                n_iter=1000, init='pca', random_state=int(seed), verbose=0)
    Z = tsne.fit_transform(X_np)
    return Z

def _save_cache(path, Z_np, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.pt', '.pth'):
        torch.save({'Z': torch.from_numpy(Z_np), 'meta': meta}, path)
    else:
        np.savez_compressed(path, Z=Z_np, meta_json=json.dumps(meta, default=lambda o: None))

def _load_cache(path):
    if not path or not os.path.exists(path):
        return None, None
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.pt', '.pth'):
        obj = torch.load(path, map_location='cpu')
        Z = obj['Z'].numpy()
        meta = obj.get('meta', {})
        return Z, meta
    else:
        obj = np.load(path, allow_pickle=True)
        Z = obj['Z']
        meta = json.loads(str(obj['meta_json'])) if 'meta_json' in obj.files else {}
        return Z, meta

class FeatureProjector:
    """
    Unified feature projector for: none | jl | pca | tsne.
    Fit once on full feature matrix; reuse across train/val/test.
    For t-SNE (non-parametric), you must cache and reuse the one-time embedding.
    """
    def __init__(self, method='jl', d_out=512, seed=0, orth=False,
                 pca_whiten=False, tsne_perplexity=30.0, tsne_lr=200.0,
                 cache_path=''):
        self.method = method
        self.d_out = int(d_out)
        self.seed = int(seed)
        self.orth = bool(orth)
        self.pca_whiten = bool(pca_whiten)
        self.tsne_perplexity = float(tsne_perplexity)
        self.tsne_lr = float(tsne_lr)
        self.cache_path = cache_path

        self.R = None         # for JL
        self.pca_meta = None  # for PCA

    def fit_transform(self, X, device='cpu'):
        X_np = _to_np(X).astype(np.float32)
        N, d_in = X_np.shape

        # Try cache first
        if self.cache_path:
            Zc, meta = _load_cache(self.cache_path)
            if Zc is not None and Zc.shape == (N, self.d_out):
                # Load JL/PCA state if present
                if meta and isinstance(meta, dict):
                    if meta.get('method') == 'jl' and 'R' in meta:
                        self.R = torch.from_numpy(np.array(meta['R'])).float()
                    elif meta.get('method') == 'pca' and 'pca_meta' in meta:
                        self.pca_meta = meta['pca_meta']
                return _to_torch(Zc, device)

        # No projection or no reduction needed
        if self.method in ('none', None) or self.d_out <= 0 or self.d_out >= d_in:
            Z_np = X_np
            meta = {'method': 'none'}
        elif self.method == 'jl':
            R = _jl_matrix(d_in, self.d_out, seed=self.seed, orth=self.orth)  # CPU
            Z_np = (torch.from_numpy(X_np) @ R).numpy()
            self.R = R
            meta = {'method': 'jl', 'R': self.R.numpy(), 'seed': self.seed, 'orth': self.orth}
        elif self.method == 'pca':
            self.pca_meta = _pca_fit(X_np, self.d_out, whiten=self.pca_whiten)
            Z_np = _pca_transform(X_np, self.pca_meta)
            meta = {'method': 'pca', 'pca_meta': self.pca_meta}
        elif self.method == 'tsne':
            Z_np = _tsne_fit_transform(X_np, self.d_out, self.tsne_perplexity, self.tsne_lr, self.seed)
            meta = {'method': 'tsne',
                    'perplexity': self.tsne_perplexity, 'lr': self.tsne_lr, 'seed': self.seed}
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        if self.cache_path:
            _save_cache(self.cache_path, Z_np, meta)
        return _to_torch(Z_np, device)

    def transform(self, X, device='cpu'):
        X_np = _to_np(X).astype(np.float32)

        if self.method in ('none', None):
            return _to_torch(X_np, device)

        if self.method == 'jl':
            if self.R is None:
                raise RuntimeError("JL projector not fitted. Call fit_transform first.")
            Z_np = (torch.from_numpy(X_np) @ self.R).numpy()
            return _to_torch(Z_np, device)

        if self.method == 'pca':
            if self.pca_meta is None:
                raise RuntimeError("PCA projector not fitted. Call fit_transform first.")
            Z_np = _pca_transform(X_np, self.pca_meta)
            return _to_torch(Z_np, device)

        if self.method == 'tsne':
            # Non-parametric: can’t transform new points; reuse cache only
            if self.cache_path:
                Zc, _ = _load_cache(self.cache_path)
                if Zc is not None:
                    return _to_torch(Zc, device)
            raise RuntimeError("t-SNE is non-parametric; use --proj-cache and fit_transform once.")

        raise ValueError(f"Unknown projection method: {self.method}")