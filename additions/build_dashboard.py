from pathlib import Path
import json
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
OUT = BASE / "outputs/01_零售经营分析"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    business = ROOT / "results/business"
    alloc = ROOT / "results/allocation"
    tables = ROOT / "analysis/reports/tables"
    dates = pd.read_csv(business / "global_daily.csv").dt.tolist()
    lookup = {d: i for i, d in enumerate(dates)}
    payload = {"dates": dates}
    for key, name, cols in [
        (
            "store",
            "mart_store_day",
            [
                "dt",
                "city_id",
                "store_id",
                "product_days",
                "oos_hours",
                "severe_days",
                "close_oos_days",
            ],
        ),
        (
            "category",
            "mart_category_day",
            [
                "dt",
                "city_id",
                "first_category_id",
                "product_days",
                "oos_hours",
                "severe_days",
            ],
        ),
        (
            "hour",
            "mart_hour_city",
            ["dt", "city_id", "hour", "observations", "oos_observations"],
        ),
    ]:
        df = pd.read_csv(business / (name + ".csv"))[cols]
        df["dt"] = df.dt.map(lookup)
        payload[key] = df.astype("int64").values.tolist()
    for key, path in [
        ("policy", alloc / "policy_summary.csv"),
        ("policyDaily", alloc / "policy_daily.csv"),
        ("ci", alloc / "test_primary_ci.csv"),
        ("persistence", business / "mart_persistence.csv"),
        ("models", tables / "model_comparison_test.csv"),
        ("backtest", tables / "backtest_summary.csv"),
    ]:
        payload[key] = json.loads(pd.read_csv(path).to_json(orient="records"))
    payload["selected"] = json.loads(
        (ROOT / "analysis/reports/tables/best_model.json").read_text()
    )["entry"]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    page = (
        (ROOT / "additions/dashboard.html")
        .read_text()
        .replace("/*PLOTLY_BUNDLE*/", get_plotlyjs())
        .replace("/*PROJECT_DATA*/", encoded)
    )
    target = OUT / "经营分析看板.html"
    target.write_text(page)
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
