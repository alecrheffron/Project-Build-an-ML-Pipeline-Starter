
import argparse
import wandb
import pandas as pd
import pytest
import os

def go(args):
    run = wandb.init(job_type="data_check")
    run.config.update(vars(args))

    # Download artifacts
    csv_path = run.use_artifact(args.csv).file()
    ref_path = run.use_artifact(args.ref).file()

    # Save local copies where tests can read them
    os.makedirs("artifacts", exist_ok=True)
    pd.read_csv(csv_path).to_csv("artifacts/data.csv", index=False)
    pd.read_csv(ref_path).to_csv("artifacts/ref_data.csv", index=False)

    # Run pytest and pass parameters via env vars
    os.environ["MIN_PRICE"] = str(args.min_price)
    os.environ["MAX_PRICE"] = str(args.max_price)
    os.environ["KL_THRESHOLD"] = str(args.kl_threshold)

    # Run tests
    tests_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "data_check", "test_data.py")
)
    retcode = pytest.main(["-vv", tests_path])
    if retcode != 0:
        raise SystemExit(retcode)

    run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--min_price", type=float, required=True)
    p.add_argument("--max_price", type=float, required=True)
    p.add_argument("--kl_threshold", type=float, required=True)
    go(p.parse_args())