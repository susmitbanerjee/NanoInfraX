from pathlib import Path
import polars as pl

from ..dataset.taxonomy import Taxonomy
from ..dataset.normalizer import normalize
from ..dataset.collectors.sqlite import collect

def run():
    root = Path(__file__).resolve().parents[4] / "datasets"
    taxonomy = Taxonomy(root/"taxonomy")
    samples=collect()

    raw = root/"raw"
    processed=root/"processed"

    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        [s.model_dump() for s in samples],
    ).write_ndjson(raw/"sqlite.jsonl")

    records = {}
    for sample in samples:
        taxonomy.validate(sample)
        text = normalize(sample.text)

        record = sample.model_dump()
        record["text"] = text

        if text in records:
            fields = ["category", "component", "cause"]
            if any(
                records[text][f] != record[f]
                for f in fields
            ):
                raise ValueError(
                    f"Conflicting labels: {text}"
                )
            continue

        records[text] = record

        df = pl.DataFrame(list(records.values()))
        df.write_parquet(processed/"errors.parquet")

        print(f"Raw samples: {len(samples)}")
        print(f"Processed samples: {df.height}")
        print(df.select("text", "category", "cause"))


if __name__ == "__main__":
    run()
