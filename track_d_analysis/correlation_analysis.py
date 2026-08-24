"""
Track D: joins Track B's entanglement metrics and Track C's efficacy/
specificity results into the final combined results table
(schema.ConceptResultRow), computes Spearman correlations between
entanglement scores and outcome variables, and plots them. This is the actual
test of the project's central hypothesis: does dictionary-space entanglement
predict unlearning difficulty?
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "track_b_entanglement"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from schema import CONCEPT_RESULT_FIELDS  # noqa: E402
from entanglement_metrics import compute_all as compute_entanglement  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts"
# Track C (run_erasure_eval.py) writes one erasure_eval__<concept>.parquet
# per concept into this directory, not one shared erasure_eval_results.parquet
# -- a crashed concept, or concepts run as separate invocations/machines,
# would otherwise clobber already-computed results the same way Track A's
# pre-build_run_tag() output collisions did. Kept separate from
# artifacts/features/ (Track A's own per-concept output directory) even
# though both are now "one file per concept" -- mixing two tracks' artifacts
# in one directory invites an accidental glob("*.parquet") picking up both.
EVAL_RESULTS_DIR = ARTIFACTS_DIR / "erasure_eval"
COMBINED_RESULTS_PATH = ARTIFACTS_DIR / "results.parquet"

ENTANGLEMENT_COLS = ["entanglement_cosine", "entanglement_cosine_paired", "entanglement_pullin_rate", "entanglement_token_overlap"]
OUTCOME_COLS = ["efficacy", "specificity_simdomain", "specificity_mmlu"]


def build_combined_results() -> pd.DataFrame:
    entanglement_df = compute_entanglement()

    eval_paths = sorted(EVAL_RESULTS_DIR.glob("erasure_eval__*.parquet"))
    if not eval_paths:
        raise FileNotFoundError(
            f"No Track C output (erasure_eval__*.parquet) in {EVAL_RESULTS_DIR}. "
            "Run track_c_erasure_eval/run_erasure_eval.py first."
        )
    eval_df = pd.concat([pd.read_parquet(p) for p in eval_paths], ignore_index=True)

    # Track B is the sole source of truth for the entanglement_* columns -- Track C's
    # ConceptResultRow rows always carry them as the schema default (None), via
    # dataclasses.asdict(). Selecting only Track C's own columns here avoids a
    # pd.merge name collision that would otherwise silently suffix both copies
    # (_x/_y) and leave the real entanglement_* values unreachable after reindex.
    eval_df = eval_df[["concept", "efficacy", "specificity_simdomain", "specificity_mmlu"]]

    combined = pd.merge(eval_df, entanglement_df, on="concept", how="outer")
    return combined.reindex(columns=CONCEPT_RESULT_FIELDS)


def compute_correlations(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ent_col in ENTANGLEMENT_COLS:
        for outcome_col in OUTCOME_COLS:
            sub = combined[[ent_col, outcome_col]].dropna()
            if len(sub) < 3:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(sub[ent_col], sub[outcome_col])
            rows.append(
                {"entanglement_metric": ent_col, "outcome": outcome_col, "spearman_rho": rho, "p_value": p, "n": len(sub)}
            )
    return pd.DataFrame(rows)


def plot_correlations(combined: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for ent_col in ENTANGLEMENT_COLS:
        for outcome_col in OUTCOME_COLS:
            sub = combined[["concept", ent_col, outcome_col]].dropna()
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(sub[ent_col], sub[outcome_col])
            for _, row in sub.iterrows():
                ax.annotate(row["concept"], (row[ent_col], row[outcome_col]), fontsize=6)
            ax.set_xlabel(ent_col)
            ax.set_ylabel(outcome_col)
            fig.tight_layout()
            fig.savefig(out_dir / f"{ent_col}__vs__{outcome_col}.png", dpi=150)
            plt.close(fig)


def main():
    combined = build_combined_results()
    combined.to_parquet(COMBINED_RESULTS_PATH, index=False)
    print(f"Wrote combined results table ({len(combined)} concepts) to {COMBINED_RESULTS_PATH}")

    correlations = compute_correlations(combined)
    correlations_path = ARTIFACTS_DIR / "correlations.csv"
    correlations.to_csv(correlations_path, index=False)
    print(correlations.to_string(index=False))
    print(f"Wrote correlation table to {correlations_path}")

    plot_correlations(combined, ARTIFACTS_DIR / "plots")


if __name__ == "__main__":
    main()
