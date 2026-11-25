import json

from pathlib import Path

class Taxonomy:
    def __init__(self, path:Path):
        self.labels = {
            field: set(
                json.loads(
                    (path/filename).read_text()
                )
            )
            for field, filename in {
                "category": "categories.json",
                "component": "components.json",
                "cause": "causes.json",
            }.items()
        }

    def validate(self, sample):
        for field, allowed in self.labels.items():
            value = getattr(sample, field)
            if value not in allowed:
                raise ValueError(f"Invalid {field}: {value}")
