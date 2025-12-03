#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aligned anisotropy computation for queries, consistent with your train/eval pipeline.

- Embeddings via SentenceTransformer.encode(..., output_value="token_embeddings")
- Keep ALL tokens (including CLS/SEP/etc.), drop ONLY padding using attention_mask
- No windowing or custom truncation; rely on model.max_seq_length as in train/eval
- FP32 only (no autocast), GPU used if available
- Optional mean-centering toggle (off by default)

Usage:
  python add_anisotropy_aligned.py \
    --ed_csv /path/to/python_top_ED_wins.csv \
    --cos_csv /path/to/python_top_Cos_wins.csv \
    --model "/gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/output/ibm-granite/granite-embedding-125m-english-CodeSearchNetCCRetrieval_ED-lr2e-5-epochs10-temperature10_full_dev" \
    --outdir ./with_anisotropy \
    --batch_size 16 \
    --device cuda \
    --mean_center false
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer


def find_text_column(df: pd.DataFrame) -> str:
    for col in ["query", "text", "query_text", "question", "qtext"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            return col
    raise ValueError(
        "Could not find a text column. Expected one of ['query','text','query_text','question','qtext'] "
        "or another object-dtype text column."
    )


@torch.no_grad()
def batched_token_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 16,
    device: str = None,
):
    """
    Uses your forked SentenceTransformer.encode(...) in token-embedding mode.
    Returns:
      embeds_batches: List[Tensor] with shapes (B, L, H) per batch
      masks_batches:  List[Tensor] with shapes (B, L) per batch
    """
    # Important: output_value="token_embeddings" returns list of 3D tensors per batch + list of masks
    embeds_batches, masks_batches = model.encode(
        texts,
        output_value="token_embeddings",
        convert_to_tensor=True,   # tensors on the chosen device
        batch_size=batch_size,
        device=device,
        normalize_embeddings=False,  # keep raw fp32 token vectors as in eval
        show_progress_bar=True,
    )
    return embeds_batches, masks_batches


@torch.no_grad()
def compute_anisotropy_from_batches(
    embeds_batches: List[torch.Tensor],
    masks_batches: List[torch.Tensor],
    mean_center: bool = False,
) -> List[float]:
    """
    Compute per-query anisotropy:
      - For each sequence i in each batch:
          X = token_embeddings[i][attention_mask[i] == 1]  # drop ONLY padding
          if X.shape[0] < 2: anisotropy = 1.0
          else s = torch.linalg.svdvals(X); aniso = s[0]^2 / sum(s^2)
    """
    anisotropy_scores: List[float] = []
    for emb3d, mask2d in zip(embeds_batches, masks_batches):
        # emb3d: (B, L, H), mask2d: (B, L)
        B, L, H = emb3d.shape
        for b in range(B):
            X = emb3d[b]  # (L, H)
            m = mask2d[b].bool()  # (L,)
            # Drop only padding (mask==0). Keep specials.
            X = X[m]

            if X.shape[0] < 2:
                anisotropy_scores.append(1.0)
                continue

            if mean_center:
                X = X - X.mean(dim=0, keepdim=True)

            # FP32, same device as X; compute singular values only
            s = torch.linalg.svdvals(X)           # (min(L,H),)
            s2 = s.pow(2)
            denom = s2.sum()
            aniso = (s2[0] / denom).item() if denom > 0 else 1.0
            anisotropy_scores.append(float(aniso))

    return anisotropy_scores


def process_csv(
    in_csv: str,
    model: SentenceTransformer,
    outdir: str,
    batch_size: int,
    device: str,
    mean_center: bool,
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    text_col = find_text_column(df)

    texts = df[text_col].astype(str).tolist()

    # Encode in batches (B, L, H) + (B, L), exactly like your eval path
    embeds_batches, masks_batches = batched_token_embeddings(
        model, texts, batch_size=batch_size, device=device
    )

    # Compute anisotropy per query in order
    scores = compute_anisotropy_from_batches(
        embeds_batches, masks_batches, mean_center=mean_center
    )
    assert len(scores) == len(texts), "Mismatch between number of queries and anisotropy scores."

    df["anisotropy"] = scores

    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(in_csv)
    root, ext = os.path.splitext(base)
    out_csv = os.path.join(outdir, f"{root}_with_anisotropy{ext}")
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv} (added 'anisotropy' column)")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed_csv", required=True, help="CSV of top ED wins")
    ap.add_argument("--cos_csv", required=True, help="CSV of top Cosine wins")
    ap.add_argument(
        "--model",
        required=True,
        help="Path/name of trained SentenceTransformer checkpoint (your ED Granite model)",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cuda:0", or "cpu" (default: auto)')
    ap.add_argument(
        "--mean_center",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Mean-center token matrix before SVD (default: false).",
    )
    args = ap.parse_args()

    # Load model in FP32; no dtype downcasting
    print(f"[Load model] {args.model}")
    model = SentenceTransformer(args.model, device=args.device)

    # Resolve device (if None, model.device decides)
    device = args.device if args.device is not None else str(model.device)

    mean_center = (args.mean_center.lower() == "true")
    if mean_center:
        print("[Info] Mean-centering is ENABLED")
    else:
        print("[Info] Mean-centering is DISABLED")

    # Process both groups
    df_ed = process_csv(
        args.ed_csv, model, args.outdir, args.batch_size, device, mean_center
    )
    df_cos = process_csv(
        args.cos_csv, model, args.outdir, args.batch_size, device, mean_center
    )

    # Group-wise averages
    mu_ed = float(df_ed["anisotropy"].mean()) if len(df_ed) else float("nan")
    mu_cos = float(df_cos["anisotropy"].mean()) if len(df_cos) else float("nan")
    print("\n=== Anisotropy Averages (aligned) ===")
    print(f"Top ED wins   : {mu_ed:.6f}")
    print(f"Top Cos wins  : {mu_cos:.6f}")


if __name__ == "__main__":
    main()

