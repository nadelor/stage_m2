"""
Unsupervised clustering of patients based on binary clinical variables.

Pipeline:
  1. Load df_final_binaire_imputed.csv
  2. Detect binary (0/1) columns automatically
  3. Build one descriptive text string per patient from those columns
  4. Embed strings with Clinical-Longformer (mean-pooled last hidden state)
  5. PCA → 50 dims (denoising before clustering)
  6. HDBSCAN on PCA embeddings (correct — not on UMAP)
  7. UMAP 2-D  }  both purely for visualisation,
     t-SNE 2-D }  coloured by HDBSCAN labels
  8. Save results CSV + two scatter plots
"""

import argparse #permet de passer des paramètres en ligne de commande
import logging #affiche les messages de progression
import os #gestion des fichiers/dossiers

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer #chargent automatiquement un modèle Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── configuration ────────────────────────────────────────────────────────────
CSV_PATH    = "df_final_binaire_imputed.csv"
MODEL_NAME  = "yikuan8/Clinical-Longformer"
BATCH_SIZE  = 64
MAX_LENGTH  = 512
PCA_COMPONENTS      = 50
UMAP_NEIGHBORS      = 30
UMAP_MIN_DIST       = 0.1
TSNE_PERPLEXITY     = 30
TSNE_ITERATIONS     = 1000
HDBSCAN_MIN_CLUSTER = 50
HDBSCAN_MIN_SAMPLES = 10
EMBEDDINGS_PATH = "Results/binary_embeddings.npy"
OUT_CSV         = "Results/binary_cluster_results.csv"
OUT_PLOT_UMAP   = "Results/binary_cluster_umap.png"
OUT_PLOT_TSNE   = "Results/binary_cluster_tsne.png"
# ─────────────────────────────────────────────────────────────────────────────


def detect_binary_columns(df: pd.DataFrame) -> list[str]:
    binary = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if set(vals).issubset({0, 1, 0.0, 1.0}) and len(vals) > 0:
            binary.append(col)
    return binary

# transforme un patient en texte descriptif du type "diabetes: 1, hypertension: 0, ...",
def build_text(row: pd.Series, cols: list[str]) -> str:
    parts = [f"{col}: {int(row[col])}" for col in cols if pd.notna(row[col])]
    return ", ".join(parts)

# les transformeurs produisent une vecteur par token, on fait une moyenne (pondérée? possible mais pas evident)
# par le masque d'attention pour obtenir un vecteur global par patient
def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device,
                batch_size=BATCH_SIZE, max_length=MAX_LENGTH) -> np.ndarray:
    all_embeddings = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
        out = model(**encoded)
        emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(emb.cpu().float().numpy())
        if (start // batch_size) % 10 == 0:
            log.info(f"  embedded {start + len(batch):>6} / {len(texts)}")
    return np.vstack(all_embeddings)


def scatter_plot(xy, labels, title, xlabel, ylabel, out_path):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_mask = labels == -1
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xy[noise_mask, 0], xy[noise_mask, 1],
               s=3, alpha=0.2, color="lightgrey", rasterized=True,
               label=f"noise (n={noise_mask.sum()})")
    for lbl in sorted(set(labels) - {-1}):
        mask = labels == lbl
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   s=5, alpha=0.55, color=cmap(lbl % 20), rasterized=True,
                   label=f"cluster {lbl} (n={mask.sum()})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved → {out_path}")


def main(args):
    os.makedirs("Results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── 1. load data ──────────────────────────────────────────────────────────
    log.info(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    log.info(f"  shape: {df.shape}")

    # ── 2. detect binary columns ──────────────────────────────────────────────
    bin_cols = detect_binary_columns(df)
    log.info(f"  binary columns ({len(bin_cols)}): {bin_cols}")

    # ── 3. build text strings ─────────────────────────────────────────────────
    log.info("Building patient text strings …")
    texts = df[bin_cols].apply(lambda row: build_text(row, bin_cols), axis=1).tolist()
    log.info(f"  example: {texts[0]}")

    # ── 4. embed (or load cached) ─────────────────────────────────────────────
    if os.path.exists(args.embeddings_path) and not args.reembed:
        log.info(f"Loading cached embeddings from {args.embeddings_path} …")
        embeddings = np.load(args.embeddings_path)
    else:
        log.info(f"Loading model '{args.model}' …")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model     = AutoModel.from_pretrained(args.model).to(device)
        log.info("Embedding patient strings …")
        embeddings = embed_texts(texts, tokenizer, model, device,
                                 batch_size=args.batch_size,
                                 max_length=args.max_length)
        np.save(args.embeddings_path, embeddings)
        log.info(f"  embeddings saved → {args.embeddings_path}")
    log.info(f"  embeddings shape: {embeddings.shape}")

    # ── 5. PCA → denoised space for clustering ────────────────────────────────
    log.info(f"PCA → {args.pca_components} components …")
    pca = PCA(n_components=args.pca_components, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    log.info(f"  explained variance: {100*explained:.1f}%")

    # ── 6. HDBSCAN on PCA embeddings ──────────────────────────────────────────
    log.info("Running HDBSCAN on PCA embeddings …")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_cluster,
        min_samples=args.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(emb_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(f"  clusters: {n_clusters}  |  noise: {n_noise} ({100*n_noise/len(labels):.1f}%)")

    # ── 7a. UMAP 2-D (visualisation only) ────────────────────────────────────
    log.info("Running UMAP for visualisation …")
    umap_2d = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
        low_memory=True,
        verbose=False,
    ).fit_transform(emb_pca)

    # ── 7b. t-SNE 2-D (visualisation only) ───────────────────────────────────
    log.info("Running t-SNE for visualisation (this may take a few minutes) …")
    tsne_2d = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        n_iter=args.tsne_iterations,
        metric="cosine",
        init="pca",
        random_state=42,
        n_jobs=-1,
    ).fit_transform(emb_pca)

    # ── 8. save results CSV ───────────────────────────────────────────────────
    result_df = df.copy()
    result_df["umap_x"]      = umap_2d[:, 0]
    result_df["umap_y"]      = umap_2d[:, 1]
    result_df["tsne_x"]      = tsne_2d[:, 0]
    result_df["tsne_y"]      = tsne_2d[:, 1]
    result_df["cluster"]     = labels
    result_df["cluster_prob"] = clusterer.probabilities_
    result_df.to_csv(args.out_csv, index=False)
    log.info(f"Results saved → {args.out_csv}")

    # ── 9. plots ──────────────────────────────────────────────────────────────
    base_title = (f"{n_clusters} clusters | noise: {n_noise} ({100*n_noise/len(labels):.1f}%)"
                  f" | mcs={args.hdbscan_min_cluster}")

    scatter_plot(umap_2d, labels,
                 title=f"UMAP — {base_title}",
                 xlabel="UMAP 1", ylabel="UMAP 2",
                 out_path=args.out_plot_umap)

    scatter_plot(tsne_2d, labels,
                 title=f"t-SNE — {base_title}",
                 xlabel="t-SNE 1", ylabel="t-SNE 2",
                 out_path=args.out_plot_tsne)

    # ── 10. cluster summary ───────────────────────────────────────────────────
    result_df["cluster_str"] = result_df["cluster"].astype(str)
    summary = result_df.groupby("cluster")[bin_cols].mean().round(3)
    log.info("\nTop-5 features per cluster:")
    for cid, row in summary.iterrows():
        top = row.sort_values(ascending=False).head(5)
        log.info(f"  Cluster {cid:>3}: {dict(top)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary-variable patient clustering")
    parser.add_argument("--csv",             default=CSV_PATH)
    parser.add_argument("--model",           default=MODEL_NAME)
    parser.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    parser.add_argument("--max-length",      type=int,   default=MAX_LENGTH)
    parser.add_argument("--pca-components",  type=int,   default=PCA_COMPONENTS)
    parser.add_argument("--umap-neighbors",  type=int,   default=UMAP_NEIGHBORS)
    parser.add_argument("--umap-min-dist",   type=float, default=UMAP_MIN_DIST)
    parser.add_argument("--tsne-perplexity", type=float, default=TSNE_PERPLEXITY)
    parser.add_argument("--tsne-iterations", type=int,   default=TSNE_ITERATIONS)
    parser.add_argument("--hdbscan-min-cluster", type=int, default=HDBSCAN_MIN_CLUSTER)
    parser.add_argument("--hdbscan-min-samples",  type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--embeddings-path", default=EMBEDDINGS_PATH)
    parser.add_argument("--reembed",   action="store_true")
    parser.add_argument("--out-csv",       default=OUT_CSV)
    parser.add_argument("--out-plot-umap", default=OUT_PLOT_UMAP)
    parser.add_argument("--out-plot-tsne", default=OUT_PLOT_TSNE)
    args = parser.parse_args()
    main(args)