from src.inference.predict import predict_single
from src.pricing.quote import generate_quote

# Example: must match your FEATURE columns (i.e., after feature build)
# For now, use one row from features_valid.parquet in a notebook/script.
example_features = {
    # ... put a dict of engineered features here ...
}

pred = predict_single(example_features)
quote = generate_quote(pred["p_claim"])

print(pred)
print(quote.to_dict())
