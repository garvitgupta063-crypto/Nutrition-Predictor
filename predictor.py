"""Load the supplied scikit-learn artifacts and make nutrition predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import joblib
import pandas as pd


FEATURES: tuple[str, ...] = (
    "total_calories",
    "fat_grams",
    "saturated_fat_grams",
    "carb_grams",
    "sugar_grams",
)


class NutritionPredictor:
    """Prediction wrapper that preserves feature names and ordering."""

    def __init__(
        self,
        model_path: str | Path = "model.pkl",
        scaler_path: str | Path = "model1.pkl",
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.model = joblib.load(base_dir / model_path)
        self.scaler = joblib.load(base_dir / scaler_path)

        artifact_features = tuple(getattr(self.scaler, "feature_names_in_", ()))
        if artifact_features and artifact_features != FEATURES:
            raise ValueError(
                "The scaler feature names do not match the expected input fields: "
                f"{artifact_features}"
            )

    def predict_one(self, values: Mapping[str, float]) -> float:
        """Return one prediction from a mapping of the five required inputs."""
        missing = [feature for feature in FEATURES if feature not in values]
        if missing:
            raise ValueError(f"Missing required input(s): {', '.join(missing)}")

        row = pd.DataFrame([[float(values[feature]) for feature in FEATURES]], columns=FEATURES)
        scaled_row = self.scaler.transform(row)
        return float(self.model.predict(scaled_row)[0])

    def predict_many(self, rows: Sequence[Mapping[str, float]]) -> list[float]:
        """Return predictions for several records."""
        return [self.predict_one(row) for row in rows]
