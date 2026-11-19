from pathlib import Path
import json
import pandas as pd
from allocation import allocate, policy_metrics, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/allocation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / "analysis/models/best_model.json").read_text())
    entry = meta["entry"]
    all_daily = []
    summaries = []
    for split, file in [
        ("valid", "predictions_valid.parquet"),
        ("test", "predictions.parquet"),
    ]:
        p = pd.read_parquet(ROOT / "analysis/data" / file)
        x = p[["dt", "store_id", "product_id", "label"]].copy()
        x["score"] = p["score_" + entry]
        assert len(x) == (700000 if split == "valid" else 350000)
        for k in [1, 3, 5, 10]:
            f = allocate(x, k)
            d, w = policy_metrics(x, f, k)
            d["split"] = split
            all_daily.append(d)
            s = d.groupby(["k", "policy"], as_index=False).mean(numeric_only=True)
            s["split"] = split
            summaries.append(s)
            assert (d.groupby("feature_date").alerts.nunique() == 1).all()
            if k == 5:
                w.to_csv(OUT / f"workloads_{split}_k5.csv", index=False)
                if split == "test":
                    ci, diff = paired_bootstrap(w)
                    ci.to_csv(OUT / "test_primary_ci.csv", index=False)
                    diff.to_csv(OUT / "test_primary_paired_diff_ci.csv", index=False)
                    # Auditable sampled task lists for first alert day; includes outcomes for retrospective review.
                    first = x.dt.min()
                    audit = x.loc[x.dt == first].copy()
                    audit["alert_date"] = audit.dt + pd.Timedelta(days=1)
                    for c in f:
                        audit[c] = f.loc[audit.index, c]
                    audit.sort_values("score", ascending=False).to_csv(
                        OUT / "first_day_retrospective_alerts.csv", index=False
                    )
            print(split, k, "done", flush=True)
    pd.concat(all_daily).to_csv(OUT / "policy_daily.csv", index=False)
    pd.concat(summaries).to_csv(OUT / "policy_summary.csv", index=False)
    (OUT / "protocol.json").write_text(
        json.dumps(
            {
                "entry": entry,
                "selection": meta["selected_by"],
                "primary_k": 5,
                "test_days": 7,
                "n_boot": 1000,
                "seed": 42,
                "tie_break": "pandas stable identity hash",
                "capacity": "hypothetical inspection scenarios, not measured staffing constraints",
                "policy_choice": "report all preregistered scenarios; no test-set optimization",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
