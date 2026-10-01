import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F


ANCHOR_KEY = "structure_branch.shapelet_dictionary.anchors"
TASKS = (
    ("AT1", "DK1"),
    ("FR1", "FR2"),
    ("FR2", "DK1"),
    ("DK1", "AT1"),
)


def load_anchor_tensor(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in packet:
        raise KeyError(f"checkpoint has no state_dict: {path}")
    state = packet["state_dict"]
    if ANCHOR_KEY not in state:
        raise KeyError(f"checkpoint has no {ANCHOR_KEY}: {path}")
    anchors = state[ANCHOR_KEY].detach().float().cpu()
    if anchors.ndim != 2:
        raise ValueError(f"anchors must be [M,D], got {tuple(anchors.shape)} in {path}")
    return anchors


def _off_diagonal_values(matrix):
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def compare_anchor_tensors(source, adapted):
    source = source.detach().float().cpu()
    adapted = adapted.detach().float().cpu()
    if source.shape != adapted.shape:
        raise ValueError(
            f"anchor shapes differ: source={tuple(source.shape)} adapted={tuple(adapted.shape)}"
        )
    source_normalized = F.normalize(source, dim=-1, eps=1e-12)
    adapted_normalized = F.normalize(adapted, dim=-1, eps=1e-12)
    row_cosine = (source_normalized * adapted_normalized).sum(dim=-1)
    difference = adapted - source
    l2 = difference.norm(dim=-1)
    source_norm = source.norm(dim=-1)
    adapted_norm = adapted.norm(dim=-1)
    relative_l2 = l2 / source_norm.clamp_min(1e-12)
    gram_source = source_normalized @ source_normalized.T
    gram_adapted = adapted_normalized @ adapted_normalized.T
    pair_source = _off_diagonal_values(gram_source)
    pair_adapted = _off_diagonal_values(gram_adapted)
    summary = {
        "mean_rowwise_cosine": float(row_cosine.mean()),
        "min_rowwise_cosine": float(row_cosine.min()),
        "max_rowwise_cosine": float(row_cosine.max()),
        "mean_relative_l2_drift": float(relative_l2.mean()),
        "pairwise_cosine_mean_before": float(pair_source.mean()),
        "pairwise_cosine_mean_after": float(pair_adapted.mean()),
        "pairwise_cosine_max_before": float(pair_source.max()),
        "pairwise_cosine_max_after": float(pair_adapted.max()),
        "gram_frobenius_drift": float((gram_adapted - gram_source).norm()),
    }
    rows = [{
        "anchor_id": index,
        "cosine": float(row_cosine[index]),
        "l2_distance": float(l2[index]),
        "relative_l2_distance": float(relative_l2[index]),
        "source_norm": float(source_norm[index]),
        "uda_norm": float(adapted_norm[index]),
    } for index in range(source.shape[0])]
    return summary, rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(checkpoint_root, output_root):
    checkpoint_root = Path(checkpoint_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    anchor_rows = []
    for source, target in TASKS:
        task = f"{source}_{target}"
        source_path = (
            checkpoint_root / "source" / f"source_{source}_seed1" / "fold_0" / "model.pt"
        )
        uda_fold = checkpoint_root / "uda" / f"{task}_seed1" / "fold_0"
        comparisons = [("best", uda_fold / "model.pt")]
        last_path = uda_fold / "checkpoint_last.pt"
        if last_path.is_file():
            comparisons.append(("last", last_path))
        source_anchors = load_anchor_tensor(source_path)
        for checkpoint_kind, adapted_path in comparisons:
            adapted_anchors = load_anchor_tensor(adapted_path)
            summary, rows = compare_anchor_tensors(source_anchors, adapted_anchors)
            summary_rows.append({
                "task": task,
                "checkpoint": checkpoint_kind,
                "source_checkpoint": str(source_path),
                "uda_checkpoint": str(adapted_path),
                **summary,
            })
            anchor_rows.extend({
                "task": task,
                "checkpoint": checkpoint_kind,
                **row,
            } for row in rows)
    _write_csv(output_root / "anchor_drift_summary.csv", summary_rows)
    _write_csv(output_root / "anchor_drift_per_anchor.csv", anchor_rows)
    lines = [
        "# V2-Clean Anchor Drift",
        "",
        "| Task | Checkpoint | Mean cosine | Min cosine | Max cosine | Mean relative L2 | Gram Frobenius drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['checkpoint']} | "
            f"{row['mean_rowwise_cosine']:.8f} | {row['min_rowwise_cosine']:.8f} | "
            f"{row['max_rowwise_cosine']:.8f} | {row['mean_relative_l2_drift']:.8f} | "
            f"{row['gram_frobenius_drift']:.8f} |"
        )
    (output_root / "anchor_drift_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    for row in summary_rows:
        print(
            "ANCHOR_DRIFT|"
            + "|".join(
                f"{key}={row[key]}" for key in (
                    "task", "checkpoint", "mean_rowwise_cosine",
                    "mean_relative_l2_drift", "gram_frobenius_drift",
                )
            )
        )
    return summary_rows


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only V2-Clean anchor drift audit")
    parser.add_argument(
        "--checkpoint-root", default="outputs/structure_proto_v2clean_4tasks_seed1",
    )
    parser.add_argument("--output-root", default="outputs/anchor_drift_v2clean_seed1")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.checkpoint_root, arguments.output_root)
