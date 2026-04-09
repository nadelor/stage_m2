"""
Sweep HDBSCAN to get ~10, 20, …, 100 clusters.

Correct order:
  - HDBSCAN runs on PCA-reduced embeddings (not on UMAP)
  - UMAP and t-SNE are computed once and used only for visualisation

Reuses Results/binary_embeddings.npy (saved by binary_cluster_embeddings.py).
Saves one PNG per target (UMAP + t-SNE side by side) → Results/cluster_sweep/
"""

import logging
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

EMBEDDINGS_PATH = "Results/binary_embeddings.npy"
OUT_DIR         = "Results/cluster_sweep"
TARGETS         = list(range(10, 101, 10))   # 10, 20, …, 100
PCA_COMPONENTS  = 50
MIN_SAMPLES     = 5                           # fixed for the sweep

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. load embeddings & PCA ──────────────────────────────────────────────────
log.info(f"Loading embeddings from {EMBEDDINGS_PATH} …")
embeddings = np.load(EMBEDDINGS_PATH)
log.info(f"  shape: {embeddings.shape}")

log.info(f"PCA → {PCA_COMPONENTS} components …")
pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
emb_pca = pca.fit_transform(embeddings)
log.info(f"  explained variance: {100*pca.explained_variance_ratio_.sum():.1f}%")

# ── 2. UMAP (once, for visualisation) ────────────────────────────────────────
log.info("Running UMAP …")
umap_2d = umap.UMAP(
    n_neighbors=30, min_dist=0.1, n_components=2,
    metric="cosine", random_state=42, low_memory=True, verbose=False,
).fit_transform(emb_pca)

# ── 3. t-SNE (once, for visualisation) ───────────────────────────────────────
log.info("Running t-SNE (may take a few minutes) …")
tsne_2d = TSNE(
    n_components=2, perplexity=30, n_iter=1000,
    metric="cosine", init="pca", random_state=42, n_jobs=-1,
).fit_transform(emb_pca)

# ── 4. sweep min_cluster_size on PCA embeddings ───────────────────────────────
log.info("Sweeping min_cluster_size on PCA embeddings …")
sweep = []   # (mcs, n_clusters, labels)
for mcs in range(3, 500):
    labels = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    ).fit_predict(emb_pca)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    sweep.append((mcs, n, labels))
    if mcs % 50 == 0:
        log.info(f"  mcs={mcs:>4}  →  {n} clusters")

# ── 5. pick closest config per target ────────────────────────────────────────
log.info("Selecting best configs …")
chosen = {}
for target in TARGETS:
    best = min(sweep, key=lambda r: abs(r[1] - target))
    chosen[target] = best
    log.info(f"  target={target:>3}  →  mcs={best[0]}  actual={best[1]}")

# ── 6. plot: UMAP + t-SNE side by side ───────────────────────────────────────
cmap = plt.cm.get_cmap("tab20")

for target, (mcs, n_actual, labels) in chosen.items():
    labels = np.array(labels)
    noise_mask = labels == -1
    cluster_ids = sorted(set(labels) - {-1})
    noise_pct = 100 * noise_mask.sum() / len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Target: {target} clusters  |  actual: {n_actual}  |  "
        f"noise: {noise_mask.sum()} ({noise_pct:.1f}%)  |  mcs={mcs}",
        fontsize=11,
    )

    for ax, xy, xlabel, ylabel, proj_name in [
        (axes[0], umap_2d, "UMAP 1",  "UMAP 2",  "UMAP"),
        (axes[1], tsne_2d, "t-SNE 1", "t-SNE 2", "t-SNE"),
    ]:
        for lbl in cluster_ids:
            mask = labels == lbl
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       s=5, alpha=0.6, color=cmap(lbl % 20), rasterized=True)
        ax.set_title(proj_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"clusters_{target:03d}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"  saved → {out_path}")

log.info("Done.")