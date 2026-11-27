from time import perf_counter
import numpy as np
import joblib

model = joblib.load("../models/baseline/model.joblib")

error = "Database connection refused"

# Warmup
for _ in range(100):
    model.predict([error])

latencies = []

for _ in range(1000):
    start = perf_counter()
    model.predict([error])
    latencies.append((perf_counter() - start) * 1000)

print("P50:", np.percentile(latencies, 50), "ms")
print("P99:", np.percentile(latencies, 99), "ms")