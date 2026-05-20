from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from app.vectorstore.chroma_store import _build_hf_embeddings
from loguru import logger

from app.core.config import settings

PLOTS_DIR = Path("data/rag/plots")
REPORT_PATH = Path("data/rag/explorer_report.json")

_SENTENCE_END = re.compile(r"[.!?\n]$")


def _connect() -> Chroma:
    embedding = _build_hf_embeddings(
        settings.resolved_rag_embedding_model,
        offline=bool(settings.rag_embeddings_offline),
        hf_home=settings.rag_hf_home,
    )
    return Chroma(
        collection_name=settings.rag_chroma_collection_name,
        embedding_function=embedding,
        persist_directory=str(settings.rag_chroma_persist_directory),
    )


def _get_all(
    vectorstore: Chroma,
) -> tuple[list[dict[str, Any]], list[str], list[list[float]]]:
    raw = vectorstore.get(include=["metadatas", "documents", "embeddings"])
    metadatas: list[dict[str, Any]] = raw.get("metadatas", []) or []
    documents: list[str] = raw.get("documents", []) or []
    embeddings_raw = raw.get("embeddings")
    embeddings: list[list[float] | None] = (
        list(embeddings_raw) if embeddings_raw is not None else []
    )
    return metadatas, documents, embeddings


def _detect_truncated(text: str) -> bool:
    return len(text) >= 790 and not bool(_SENTENCE_END.search(text.rstrip()))


def inspect() -> dict[str, Any]:
    logger.info("Connecting to ChromaDB for inspection...")
    vectorstore = _connect()
    metadatas, documents, embeddings = _get_all(vectorstore)

    logger.info(f"Total chunks: {len(documents)}")

    artifact_counter: Counter[str] = Counter()
    chunk_lengths: list[int] = []
    truncated: list[dict[str, Any]] = []
    docs_per_artifact: dict[str, set[str]] = defaultdict(set)
    unique_metadata_keys: set[str] = set()
    metadata_samples: dict[str, set[Any]] = defaultdict(set)

    for meta, doc in zip(metadatas, documents):
        aid = meta.get("artifact_id", "unknown")
        artifact_counter[aid] += 1
        chunk_lengths.append(len(doc))
        docs_per_artifact[aid].add(meta.get("content_hash", ""))

        for k, v in meta.items():
            unique_metadata_keys.add(k)
            metadata_samples[k].add(str(v)[:80])

        if _detect_truncated(doc):
            truncated.append(
                {
                    "chunk_index": meta.get("chunk_index"),
                    "artifact_id": aid,
                    "content_hash": meta.get("content_hash"),
                    "length": len(doc),
                }
            )

    n_embeddings = len([e for e in embeddings if e is not None])

    report: dict[str, Any] = {
        "total_chunks": len(documents),
        "total_embeddings": n_embeddings,
        "embedded_ratio": round(n_embeddings / len(documents), 3) if documents else 0,
        "artifact_count": len(artifact_counter),
        "chunks_per_artifact": dict(artifact_counter.most_common()),
        "documents_per_artifact": {
            aid: len(hashes) for aid, hashes in docs_per_artifact.items()
        },
        "chunk_length": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "mean": round(sum(chunk_lengths) / len(chunk_lengths), 1)
            if chunk_lengths
            else 0,
            "histogram_bins": _histogram(chunk_lengths, 10),
        },
        "truncated_chunks": {
            "count": len(truncated),
            "ratio": round(len(truncated) / len(documents), 3) if documents else 0,
            "per_artifact": dict(
                Counter(t["artifact_id"] for t in truncated).most_common()
            ),
        },
        "metadata_keys": sorted(unique_metadata_keys),
        "metadata_samples": {k: list(v)[:3] for k, v in metadata_samples.items()},
    }

    logger.info(
        f"Inspection complete — {report['total_chunks']} chunks across {report['artifact_count']} artifact types"
    )
    return report


def _histogram(values: list[int], bins: int) -> list[dict[str, int]]:
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [{"bin_start": min_v, "bin_end": max_v + 1, "count": len(values)}]
    step = (max_v - min_v) / bins
    edges = [int(min_v + i * step) for i in range(bins)] + [max_v + 1]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - min_v) / step), bins - 1)
        counts[idx] += 1
    return [
        {"bin_start": edges[i], "bin_end": edges[i + 1], "count": counts[i]}
        for i in range(bins)
    ]


def save_report(report: dict[str, Any]) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Report saved to {REPORT_PATH}")
    return REPORT_PATH


def generate_plots(report: dict[str, Any]) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

    # 1 — Chunks per artifact (horizontal bar)
    artifacts = list(report["chunks_per_artifact"].keys())
    counts = list(report["chunks_per_artifact"].values())
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(artifacts, counts, color="#4A90D9", edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(c),
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Chunks")
    ax.set_title("Chunks per Artifact Type")
    ax.margins(x=0.15)
    fig.tight_layout()
    p = PLOTS_DIR / "chunks_per_artifact.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # 2 — Chunk length histogram
    bins = report["chunk_length"]["histogram_bins"]
    if bins:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{b['bin_start']}-{b['bin_end']}" for b in bins]
        vals = [b["count"] for b in bins]
        colors = [
            "#E74C3C" if "bin_start" in b and b["bin_start"] >= 790 else "#2ECC71"
            for b in bins
        ]
        ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.axvline(
            x=len(bins) - 1.5,
            color="red",
            linestyle="--",
            linewidth=1,
            label="800-char truncation boundary",
        )
        ax.legend(fontsize=8)
        ax.set_xlabel("Chunk length (chars)")
        ax.set_ylabel("Count")
        ax.set_title("Chunk Length Distribution")
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_fontsize(7)
        fig.tight_layout()
        p2 = PLOTS_DIR / "chunk_length_distribution.png"
        fig.savefig(p2)
        plt.close(fig)
        paths.append(p2)

    # 3 — Truncation rate per artifact
    trunc = report["truncated_chunks"]["per_artifact"]
    all_aids = set(artifacts)
    trunc_rates: dict[str, float] = {}
    for aid in all_aids:
        total = report["chunks_per_artifact"].get(aid, 0)
        tr = trunc.get(aid, 0)
        trunc_rates[aid] = round(tr / total * 100, 1) if total else 0.0
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_aids = sorted(trunc_rates.keys(), key=lambda a: trunc_rates[a], reverse=True)
    vals = [trunc_rates[a] for a in sorted_aids]
    colors_bar = ["#E74C3C" if v > 0 else "#95A5A6" for v in vals]
    ax.barh(sorted_aids, vals, color=colors_bar, edgecolor="white")
    for bar, v in zip(ax.containers[0], vals):
        if v > 0:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{v}%",
                va="center",
                fontsize=9,
            )
    ax.set_xlabel("Truncated chunks (%)")
    ax.set_title("Truncation Rate per Artifact Type")
    ax.margins(x=0.15)
    fig.tight_layout()
    p3 = PLOTS_DIR / "truncation_rate.png"
    fig.savefig(p3)
    plt.close(fig)
    paths.append(p3)

    # 4 — Document vs chunk count (stacked comparison)
    doc_counts = report.get("documents_per_artifact", {})
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(artifacts))
    w = 0.35
    chunk_vals = [report["chunks_per_artifact"].get(a, 0) for a in artifacts]
    doc_vals = [doc_counts.get(a, 0) for a in artifacts]
    ax.bar([i - w / 2 for i in x], chunk_vals, w, label="Chunks", color="#4A90D9")
    ax.bar([i + w / 2 for i in x], doc_vals, w, label="Documents", color="#2ECC71")
    ax.set_xticks(list(x))
    ax.set_xticklabels(artifacts, rotation=30, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Chunks vs Source Documents per Artifact")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p4 = PLOTS_DIR / "chunks_vs_documents.png"
    fig.savefig(p4)
    plt.close(fig)
    paths.append(p4)

    logger.info(f"Generated {len(paths)} plots in {PLOTS_DIR}")
    return paths


def run() -> dict[str, Any]:
    report = inspect()
    save_report(report)
    generate_plots(report)
    return report


if __name__ == "__main__":
    run()
