from src.inference.predict import load_model
from src.features.runtime import build_features_from_raw

artifact = load_model()

raw_policy = {
    "subscription_length": 9.3,
    "vehicle_age": 1.2,
    "customer_age": 41,
    "region_code": "C8",
    "region_density": 8794,
    "segment": "C2",
    "model": "M4",
    "fuel_type": "Diesel",
    "engine_type": "E2",
    "max_torque": "250Nm@2750rpm",
    "max_power": "100.6bhp@6000rpm",
    "airbags": 2,
    "is_esc": "Yes",
    "is_tpms": "No",
    "rear_brakes_type": "Drum",
    "transmission_type": "Manual",
    "steering_type": "Power",
    "ncap_rating": 3,
}

built = build_features_from_raw(raw_policy, artifact.feature_columns)
p_claim = artifact.model.predict_proba(built.features)[:, 1][0]

print("Warnings:", built.warnings)
print("p_claim:", float(p_claim))
