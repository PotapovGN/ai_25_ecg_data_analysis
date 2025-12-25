import pickle
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Поставил, чтобы игнорировать несоответствие версий sklearn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Патология
with open("models/patology_logistic_model.pkl", "rb") as f:
    pathology_data = pickle.load(f)
    pathology_model = pathology_data["model"]
    pathology_feature_order = pathology_data.get("feature_order", None)
    pathology_threshold = 0.4  # Порог для патологии

# Аритмия
with open("models/arrhythmia_logistic_model.pkl", "rb") as f:
    arrhythmia_data = pickle.load(f)
    arrhythmia_model = arrhythmia_data["model"]
    arrhythmia_feature_order = arrhythmia_data.get("feature_order", None)
    arrhythmia_threshold = arrhythmia_data.get("default_threshold", 0.2)

# Инфаркт
with open("models/infarction_logistic_model.pkl", "rb") as f:
    infarction_data = pickle.load(f)
    infarction_model = infarction_data["model"]
    infarction_feature_order = infarction_data.get("feature_order", None)
    infarction_threshold = infarction_data.get("default_threshold", 0.2)

# Scaler
with open("models/scaler_with_columns.pkl", "rb") as f:
    scaler_data = pickle.load(f)

scaler = scaler_data["scaler"]

# Добавление категориальных фичей
numeric_columns = list(scaler.feature_names_in_)
categorical_columns = ['heart_axis_norm', 'V1_pathological_Q']
trained_columns_order = numeric_columns + categorical_columns