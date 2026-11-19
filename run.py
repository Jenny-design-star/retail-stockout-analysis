"""Single entry point. Run from the project directory with its virtualenv Python."""

from pathlib import Path
import os, sys, subprocess, argparse, json, hashlib, urllib.request

ROOT = Path(__file__).resolve().parent


def download():
    p = json.loads((ROOT / "provenance.json").read_text())
    rev = p["dataset"]["revision"]
    for name, record in p["data"].items():
        target = ROOT / "analysis/data/raw" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f"https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/{rev}/data/{name}"
            tmp = target.with_suffix(".download")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        print(name, "verified", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "download",
            "warehouse",
            "checks",
            "backtest",
            "train",
            "report",
            "business",
            "allocation",
            "dashboard",
            "delivery",
            "predict",
            "test",
        ],
    )
    args = p.parse_args()
    stage = args.stage
    if stage == "download":
        download()
        return
    env = os.environ.copy()
    env.update(
        {
            "MPLCONFIGDIR": str(ROOT / ".mplcache"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "6",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib")
    (ROOT / "logs").mkdir(exist_ok=True)
    if stage in ["warehouse", "checks", "backtest", "train", "report", "predict"]:
        cwd = ROOT / "analysis"
        if stage in ["backtest", "train"]:
            module = "src.backtest" if stage == "backtest" else "src.modeling"
            cmd = [
                sys.executable,
                "-c",
                f"from src import modeling as m; m.LGB_PARAMS['n_jobs']=6; import {module} as entry; entry.main()",
            ]
        elif stage == "predict":
            cmd = [
                sys.executable,
                "-m",
                "src.predict",
                "--date",
                "2024-07-02",
                "--top-k",
                "5000",
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "src." + ("run_sql" if stage == "warehouse" else stage),
            ]
    elif stage == "test":
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=ROOT / "analysis",
            env=env,
            check=True,
        )
        cmd = [sys.executable, "-m", "pytest", "-q", "additions/test_allocation.py"]
        cwd = ROOT
    else:
        file = {
            "business": "build_business.py",
            "allocation": "run_allocation.py",
            "dashboard": "build_dashboard.py",
            "delivery": "build_delivery.py",
        }[stage]
        cwd = ROOT
        cmd = [sys.executable, str(ROOT / "additions" / file)]
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
