import sqlite3
from contextlib import closing

from ..schema import TrainingSample

def collect() -> list[TrainingSample]:
    samples = []

    cases = [
        (
            "SELECT * FROM missing_table",
            "SCHEMA_ERROR",
            "TABLE_NOT_FOUND",
        ),
        (
            "INSERT INTO users VALUES (1)",
            "DATA_INTEGRITY",
            "UNIQUE_CONSTRAINT_VIOLATION"
        ),
        (
            "INSERT INTO users VALUES (NULL)",
            "DATA_INTEGRITY",
            "NOT_NULL_VIOLATION"
        )
    ]

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (id INT NOT NULL UNIQUE)"
        )
        conn.execute("INSERT INTO users VALUES (1)")

        for query, category, cause in cases:
            try:
                conn.execute(query)
            except sqlite3.Error as error:
                samples.append(
                    TrainingSample(
                        text=f"SQLite error: {error}",
                        category=category,
                        component="DATABASE",
                        cause=cause,
                        source = "sqlite_fault_injection",
                        source_group=f"sqlite_{cause.lower()}",
                        synthetic=False
                ))

    return samples