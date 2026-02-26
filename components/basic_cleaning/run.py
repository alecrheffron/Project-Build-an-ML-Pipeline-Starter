import argparse
import pandas as pd
import wandb


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    artifact = run.use_artifact(args.input_artifact)
    input_path = artifact.file()

    df = pd.read_csv(input_path)

    # Basic cleaning consistent with the project notebook
    # 1) price to numeric (strip $ and commas)
    df["price"] = (
        df["price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # 2) last_review to datetime
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # 3) drop rows with invalid price + clip to bounds
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= args.min_price) & (df["price"] <= args.max_price)]

    # Save cleaned
    out_path = "clean_sample.csv"
    df.to_csv(out_path, index=False)

    out_art = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    out_art.add_file(out_path)
    run.log_artifact(out_art)
    run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_artifact", required=True)
    p.add_argument("--output_artifact", required=True)
    p.add_argument("--output_type", required=True)
    p.add_argument("--output_description", required=True)
    p.add_argument("--min_price", type=float, required=True)
    p.add_argument("--max_price", type=float, required=True)
    go(p.parse_args())