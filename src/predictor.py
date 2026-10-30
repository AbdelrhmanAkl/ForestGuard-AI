
import json
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO_ID = "AbdelrahmanAkl/forestguard-ai-model"

MODEL_PATH = PROJECT_ROOT / "models" / "forest_cover_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def resolve_artifact(local_path, filename):
    """Use the local artifact if available; otherwise download it from Hugging Face."""
    if local_path.exists():
        return local_path

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        repo_type="model",
    )

    return Path(downloaded_path)


def load_artifacts():
    """Load the trained model and its metadata."""
    resolved_model_path = resolve_artifact(
        MODEL_PATH,
        "forest_cover_model.joblib",
    )

    resolved_metadata_path = resolve_artifact(
        METADATA_PATH,
        "model_metadata.json",
    )

    with resolved_metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(resolved_model_path)

    expected_features = metadata["feature_names"]

    if len(expected_features) != model.n_features_in_:
        raise ValueError("Metadata feature count does not match the model.")

    return model, metadata


def predict_cover_type(model, metadata, input_features):
    """
    Predict forest cover type from a dictionary of input features.

    Returns:
        dict: Predicted class, class probabilities, and confidence.
    """
    feature_names = metadata["feature_names"]

    missing_features = set(feature_names) - set(input_features)

    if missing_features:
        raise ValueError(f"Missing features: {sorted(missing_features)}")

    ordered_input = {
        feature: input_features[feature]
        for feature in feature_names
    }

    input_df = pd.DataFrame([ordered_input], columns=feature_names)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    class_probabilities = {
        str(int(class_label)): float(probability)
        for class_label, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    return {
        "predicted_class": prediction,
        "confidence": float(max(probabilities)),
        "class_probabilities": class_probabilities,
    }
