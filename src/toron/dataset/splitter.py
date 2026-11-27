from pathlib import Path
from random import Random

import polars as pl

def split(df: pl.DataFrame):
    rng = Random(42)
    assignments = {}

    for cause in sorted(df["cause"].unique().to_list()):
        groups = (
            df.filter(pl.col("cause") == cause)
            ["source_group"].unique().sort().to_list()
        )
        if len(groups) < 3:
            raise ValueError(f"Insufficient groups: {cause}")
        rng.shuffle(groups)
        assignments[groups[0]] = "test"
        assignments[groups[1]] = "validation"

        for group in groups[2:]:
            assignments[group] = "train"

    df = df.with_columns(
        pl.col("source_group").replace_strict(assignments).alias("split")
    )

    path = Path("../datasets/processed")
    path.mkdir(parents=True, exist_ok=True)

    for name in ["train", "validation", "test"]:
        subset = df.filter(pl.col("split") == name)
        subset.write_parquet(path/ f"{name}.parquet")
        print(name, subset.height)

if __name__ == "__main__":
    df = pl.read_parquet(
        "../datasets/synthetic/errors.parquet"
    )
    split(df)