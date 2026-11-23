"""Shared CPU inference without changing the frame used by other models."""

import io
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import torch


BASE_DIR = Path(__file__).resolve().parent
MODEL_TYPES = ("TabM", "CatBoost", "LightGBM")
PREDICTION_THRESHOLD = 0.37


def load_model_on_cpu(path):
    """Map tensors embedded in a joblib pickle to CPU during loading."""
    def load_storage(data):
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

    with patch("torch.storage._load_from_bytes", load_storage):
        model = joblib.load(path)

    model.set_params(device="cpu")
    # pytabkit's fitted TabM interface stores device_ independently of the
    # sklearn constructor parameter. set_params alone does not update it.
    interfaces = [model.alg_interface_]
    while interfaces:
        interface = interfaces.pop()
        interfaces.extend(getattr(interface, "sub_split_interfaces", []))
        if hasattr(interface, "device_"):
            interface.device_ = "cpu"
        for attribute in ("model_", "tfm_"):
            module = getattr(interface, attribute, None)
            if isinstance(module, torch.nn.Module):
                module.to("cpu")
    return model


def load_model_bundle(experiment):
    models = []
    for model_type in MODEL_TYPES:
        for fold in range(1, 6):
            path = BASE_DIR / "models" / f"{model_type}_{experiment}_fold_{fold}.pkl"
            if model_type == "TabM":
                model = load_model_on_cpu(path)
            else:
                model = joblib.load(path)
            models.append(model)
    meta_model = joblib.load(BASE_DIR / "models" / "meta_model_logistic_regression.pkl")
    return models, meta_model


def catboost_frame(model, frame):
    """Restore categorical strings even if another caller supplied pandas categories."""
    result = frame.copy()
    for index in model.get_cat_feature_indices():
        column = result.columns[index]
        result[column] = result[column].astype(object).fillna("unknown").map(str)
    return result


def predict_ensemble(models, meta_model, frame):
    if frame.empty:
        raise ValueError("Нет строк для получения прогноза.")
    if len(models) != 15:
        raise ValueError("Для прогноза необходимы все 15 моделей ансамбля.")

    probabilities = {}
    for group, model_type in enumerate(MODEL_TYPES):
        folds = []
        for model in models[group * 5:(group + 1) * 5]:
            # LightGBM requires category dtype; TabM and CatBoost need their own
            # unmodified inputs. Never mutate the shared frame in this loop.
            if model_type == "CatBoost":
                model_input = catboost_frame(model, frame)
            else:
                model_input = frame.copy()
                if model_type == "LightGBM":
                    for column in model_input.select_dtypes(include=["object", "string"]):
                        model_input[column] = model_input[column].astype("category")
            folds.append(model.predict_proba(model_input)[:, 1])
        probabilities[model_type] = np.mean(folds, axis=0)

    meta_input = pd.DataFrame(probabilities, columns=MODEL_TYPES)
    if not hasattr(meta_model, "feature_names_in_"):
        meta_input = meta_input.to_numpy()
    return meta_model.predict_proba(meta_input)[:, 1]
