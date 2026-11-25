from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainingSample:
    text: str
    category: str
    component: str
    cause: str
    source: str
    source_group: str
    synthetic: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for field_name in (
            "text",
            "category",
            "component",
            "cause",
            "source",
            "source_group",
            "schema_version",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")

        if not self.text:
            raise ValueError("text must contain at least one character")
        if not isinstance(self.synthetic, bool):
            raise TypeError("synthetic must be a boolean")

    def model_dump(self) -> dict[str, object]:
        return asdict(self)
