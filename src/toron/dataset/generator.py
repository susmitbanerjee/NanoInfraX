from pathlib import Path
import polars as pl
from ..dataset.schema import TrainingSample

TEMPLATES = {
    "CONNECTION_REFUSED": (
        "CONNECTIVITY",
        "NETWORK",
        [
            "Connection refused",
            "ECONNREFUSED",
            "Remote host refused the connection"
        ]
    ),
    "DEADLOCK": (
        "CONCURRENCY",
        "DATABASE",
        [
            "Deadlock detected",
            "Transaction aborted due to deadlock",
            "Circular lock dependency detected"
        ]
    ),
    "DISK_FULL": (
        "RESOURCE_EXHAUSTION",
        "FILESYSTEM",
        [
            "No space left on device",
            "ENOSPC",
            "Disk capacity exhausted"
        ]
    )
}

PREFIXES = ["", "ERROR: ", "WARN: ", "Exception: "]

def generate() -> list[TrainingSample]:
    samples = []
    for cause, (category, component, templates) in TEMPLATES.items():
        for idx, template in enumerate(templates):
            for prefix in PREFIXES:
                samples.append(TrainingSample(
                    text=f"{prefix}{template}",
                    category=category,
                    component=component,
                    cause=cause,
                    source="template_generator",
                    source_group=f"{cause}_{idx}",
                    synthetic=True
                )
    )
    return samples

if __name__ == "__main__":
    samples = generate()
    df = pl.DataFrame([s.model_dump() for s in samples])
    path = Path("../datasets/synthetic")
    path.mkdir(parents=True, exist_ok=True)

    df.write_parquet(path/"errors.parquet")
    print(df.group_by("cause").len)
    print(df.select("text"))
