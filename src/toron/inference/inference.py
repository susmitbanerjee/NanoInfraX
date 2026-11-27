import joblib

model = joblib.load("../models/baseline/model.joblib")

errors = [
    "Connection refused by remote server",
    "Database transaction deadlock detected",
    "No space available on disk"
]

for error in errors:
    prediction = model.predict([error])[0]
    probability = model.predict_proba([error])[0].max()

    print(prediction, round(probability, 4))