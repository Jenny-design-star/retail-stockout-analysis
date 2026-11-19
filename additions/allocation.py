"""New matched-budget alert allocation. Labels never enter the allocation function."""

from __future__ import annotations
import numpy as np
import pandas as pd

POLICIES = ("global_matched", "per_store_top_n", "one_per_store_then_global")


def allocate(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """Same daily TOTAL budget for all policies; only per_store_top_n is a store cap.

    Ties use a stable hash of identities, independent of labels and input row order.
    Input dt is the FEATURE date D; actual work/label date is D+1.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    keys = ["dt", "store_id", "product_id"]
    x = frame[keys + ["score"]].copy().reset_index(drop=True)
    if x.empty or x[keys].isna().any().any() or x.duplicated(keys).any():
        raise ValueError("nonempty unique complete identities required")
    if not np.isfinite(x.score).all():
        raise ValueError("scores must be finite")
    x["_tie"] = pd.util.hash_pandas_object(x[keys], index=False).to_numpy()
    out = pd.DataFrame(False, index=x.index, columns=POLICIES)
    for _, day in x.groupby("dt", sort=True):
        ranked = day.sort_values(
            ["score", "_tie", "store_id", "product_id"],
            ascending=[False, True, True, True],
        )
        rank = ranked.groupby("store_id", sort=False).cumcount()
        local = ranked.index[rank < k]
        budget = len(local)
        floor = ranked.index[rank == 0]
        residual = ranked.index[~ranked.index.isin(floor)][: budget - len(floor)]
        out.loc[ranked.index[:budget], "global_matched"] = True
        out.loc[local, "per_store_top_n"] = True
        out.loc[floor.union(residual), "one_per_store_then_global"] = True
    return out


def policy_metrics(
    frame: pd.DataFrame, flags: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = []
    workloads = []
    for policy in POLICIES:
        x = frame[["dt", "store_id", "label"]].reset_index(drop=True).copy()
        x["alerts"] = flags[policy].astype(int)
        x["tp"] = x.alerts * x.label
        store = x.groupby(["dt", "store_id"], as_index=False).agg(
            alerts=("alerts", "sum"),
            tp=("tp", "sum"),
            positives=("label", "sum"),
            items=("label", "size"),
        )
        store["policy"] = policy
        store["k"] = k
        workloads.append(store)
        for dt, g in store.groupby("dt"):
            a = int(g.alerts.sum())
            tp = int(g.tp.sum())
            pos = int(g.positives.sum())
            daily.append(
                dict(
                    k=k,
                    policy=policy,
                    feature_date=dt,
                    alert_date=pd.Timestamp(dt) + pd.Timedelta(days=1),
                    alerts=a,
                    tp=tp,
                    false_alerts=a - tp,
                    positives=pos,
                    precision=tp / a,
                    recall=tp / pos if pos else np.nan,
                    stores=len(g),
                    stores_covered=int((g.alerts > 0).sum()),
                    store_coverage=float((g.alerts > 0).mean()),
                    stores_over_k=int((g.alerts > k).sum()),
                    p95_workload=float(g.alerts.quantile(0.95)),
                    max_workload=int(g.alerts.max()),
                )
            )
    return pd.DataFrame(daily), pd.concat(workloads, ignore_index=True)


def paired_bootstrap(workloads: pd.DataFrame, n_boot: int = 1000, seed: int = 42):
    """Resample stores, retaining all days. Freeze lists, do not rerank test labels.

    Estimates uncertainty across this population of stores, conditional on one week,
    this fitted model, and these alert lists. Does NOT estimate future-week uncertainty.
    """
    stores = np.sort(workloads.store_id.unique())
    days = np.sort(workloads.dt.unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(stores), np.full(len(stores), 1 / len(stores)), size=n_boot
    )
    draws = {}
    rows = []
    for policy in POLICIES:
        g = workloads[workloads.policy == policy]
        arrays = {}
        for c in ["alerts", "tp", "positives"]:
            mat = (
                g.pivot(index="store_id", columns="dt", values=c)
                .reindex(index=stores, columns=days)
                .fillna(0)
                .to_numpy(float)
            )
            arrays[c] = weights @ mat

        def ratio(denominator):
            a = arrays[denominator]
            r = np.divide(arrays["tp"], a, out=np.full_like(a, np.nan), where=a > 0)
            return np.mean(r, axis=1)  # invalid draw if ANY day has no denominator

        draws[policy] = {
            "precision": ratio("alerts"),
            "recall": ratio("positives"),
            "tp_per_day": np.mean(arrays["tp"], axis=1),
        }
        for metric, vals in draws[policy].items():
            lo, hi = np.nanquantile(vals, [0.025, 0.975])
            rows.append(
                dict(
                    policy=policy,
                    metric=metric,
                    lo=lo,
                    hi=hi,
                    n_boot=n_boot,
                    n_valid=int(np.isfinite(vals).sum()),
                )
            )
    diffs = []
    for policy in POLICIES[1:]:
        for metric in ["precision", "recall", "tp_per_day"]:
            vals = draws[policy][metric] - draws["global_matched"][metric]
            lo, hi = np.nanquantile(vals, [0.025, 0.975])
            diffs.append(
                dict(
                    policy=policy,
                    reference="global_matched",
                    metric=metric,
                    lo=lo,
                    hi=hi,
                    n_boot=n_boot,
                    n_valid=int(np.isfinite(vals).sum()),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(diffs)
