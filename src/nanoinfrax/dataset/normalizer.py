import re

UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}\b",
    re.IGNORECASE
)

def normalize(text:str) -> str:
    text = UUID_PATTERN.sub("[UUID]", text)
    text = re.sub(r"\s+", "", text)

    return text.strip()