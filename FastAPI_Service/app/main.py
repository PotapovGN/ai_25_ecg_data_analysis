import time
import numpy as np
import pandas as pd
import wfdb
import tempfile
import os
import shutil
import ast

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException

from app.database import SessionLocal, RequestHistory
from app.inference import (
    pathology_model,
    arrhythmia_model,
    infarction_model,
    scaler,
    trained_columns_order,
    pathology_threshold,
    arrhythmia_threshold,
    infarction_threshold,
    pathology_feature_order,
    arrhythmia_feature_order,
    infarction_feature_order
)
from app.feature_extraction import extract_features_for_model

# Инициализация приложения FastAPI + токен
app = FastAPI(title="ECG ML Service")
SECRET_TOKEN = "token_poken"

def predict_with_model(model, features_df, feature_order, threshold):
    if feature_order is not None:
        features_ordered = features_df.reindex(columns=feature_order, fill_value=0.0).values
    else:
        features_ordered = features_df.values
    
    proba = model.predict_proba(features_ordered)[0]
    prediction = int((proba[1] > threshold))
    
    return {
        'prediction': prediction,
        'probabilities': {
            'class_1_percent': round(proba[1] * 100, 1),
            'class_0_percent': round(proba[0] * 100, 1)
        }
    }

# Чтение метаинформации пациента из CSV файла
def parse_meta_csv(upload_file):
    df = pd.read_csv(upload_file.file)
    age = int(df.loc[0, "age"])
    
    # sex: 0 = male, 1 = female
    sex_raw = df.loc[0, "sex"]
    sex = "M" if sex_raw == 0 else "F"

    heart_axis = df.loc[0, "heart_axis"]

    # если heart_axis пуст — ставим normal
    if pd.isna(heart_axis) or heart_axis == "":
        heart_axis = "normal"

    return {
        "age": age,
        "sex": sex,
        "heart_axis": heart_axis
    }

# POST/forward
@app.post("/forward")
async def forward(
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...),
    meta_csv: Optional[UploadFile] = File(None),

    age: Optional[int] = Header(default=None),
    sex: Optional[str] = Header(default=None),
    heart_axis: Optional[str] = Header(default=None),
):
    """
    Принимает одну пару ЭКГ dat + hea одного пациента (lr - низкое разрешение, hr - высокое разрешение).
    Метаданные пациента передаются через отдельный CSV файл, либо через headers.
    """

    start_time = time.time()
    status = "failed"

    # Метаданные пациента
    if meta_csv:
        meta = parse_meta_csv(meta_csv)
    else:
        if age is None or sex is None or heart_axis is None:
            raise HTTPException(status_code=400, detail="bad request")

        meta = {
            "age": age,
            "sex": sex,
            "heart_axis": heart_axis
        }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Сохранение .dat файла
            dat_path = os.path.join(tmpdir, dat_file.filename)
            with open(dat_path, "wb") as f:
                shutil.copyfileobj(dat_file.file, f)
            
            # Сохранение .hea файла
            hea_path = os.path.join(tmpdir, hea_file.filename)
            with open(hea_path, "wb") as f:
                shutil.copyfileobj(hea_file.file, f)
            
            # Обработка файла
            record_name = os.path.splitext(dat_path)[0]

            # Чтение ЭКГ через wfdb
            record = wfdb.rdrecord(record_name)
            ecg_record = {
                lead: record.p_signal[:, i]
                for i, lead in enumerate(record.sig_name)
            }

            fs = record.fs

            # Извлечение признаков
            features = extract_features_for_model(
                ecg_record=ecg_record,
                fs=fs,
                patient_meta=meta,
                scaler=scaler,
                trained_columns_order=trained_columns_order
            )
            
            # Определение разрешения ЭКГ
            dat_filename = os.path.basename(dat_path)
            if 'lr' in dat_filename.lower():
                resolution_label = "Low Resolution"
            elif 'hr' in dat_filename.lower():
                resolution_label = "High Resolution"
            else:
                resolution_label = "Unknown Resolution"
            
            # Прогон через модели с получением вероятностей
            pathology_result = predict_with_model(
                pathology_model, features, pathology_feature_order, pathology_threshold
            )
            arrhythmia_result = predict_with_model(
                arrhythmia_model, features, arrhythmia_feature_order, arrhythmia_threshold
            )
            infarction_result = predict_with_model(
                infarction_model, features, infarction_feature_order, infarction_threshold
            )

            # Формирование результата
            result = {
                "label": resolution_label,
                "ecg_record": {
                    "age": meta["age"],
                    "sex": meta["sex"]
                },
                "prediction": {
                    "pathology": {
                        "prediction": pathology_result["prediction"],
                        "probabilities": {
                            "pathology_percent": pathology_result["probabilities"]["class_1_percent"],
                            "normal_percent": pathology_result["probabilities"]["class_0_percent"]
                        }
                    },
                    "arrhythmia": {
                        "prediction": arrhythmia_result["prediction"],
                        "probabilities": {
                            "arrhythmia_percent": arrhythmia_result["probabilities"]["class_1_percent"],
                            "normal_rhythm_percent": arrhythmia_result["probabilities"]["class_0_percent"]
                        }
                    },
                    "infarction": {
                        "prediction": infarction_result["prediction"],
                        "probabilities": {
                            "infarction_risk_percent": infarction_result["probabilities"]["class_1_percent"],
                            "no_risk_percent": infarction_result["probabilities"]["class_0_percent"]
                        }
                    }
                }
            }

        status = "success"

        # Метаданные
        metadata_dict = {
            "resolution": resolution_label,
            "age": meta["age"],
            "sex": meta["sex"]
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )

    finally:
        elapsed = time.time() - start_time
        db = SessionLocal()

        if 'metadata_dict' not in locals():
            metadata_dict = {
                "resolution": "Unknown",
                "age": meta.get("age") if 'meta' in locals() else None,
                "sex": meta.get("sex") if 'meta' in locals() else None
            }

        db.add(
            RequestHistory(
                processing_time=elapsed,
                status=status,
                request_metadata=str(metadata_dict)
            )
        )
        db.commit()
        db.close()

    return result

# GET/history
@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(RequestHistory).all()
    db.close()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "processing_time": r.processing_time,
            "status": r.status
        }
        for r in records
    ]

# DELETE/history
@app.delete("/history")
def delete_history(token: str = Header(...)):
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")

    db = SessionLocal()
    db.query(RequestHistory).delete()
    db.commit()
    db.close()
    return {"message": "history deleted"}

# GET/stats
@app.get("/stats")
def get_stats():
    db = SessionLocal()
    records = db.query(RequestHistory).all()
    db.close()

    if not records:
        return {"message": "No data yet"}

    times = [r.processing_time for r in records]
    
    # Все метаданные
    all_files = []
    all_ages = []
    success_count = 0
    
    for r in records:
        if r.request_metadata and r.request_metadata.strip():
            try:
                meta = ast.literal_eval(r.request_metadata)
                all_files.append(meta.get("files_count", 0))
                all_ages.append(meta.get("age", 0))
            except (ValueError, SyntaxError):
                pass
        
        # Кол-во успешных запросов
        if r.status == "success":
            success_count += 1

    return {
        "time_stats": {
            "mean": round(float(np.mean(times)), 3) if times else 0,
            "p50": round(float(np.percentile(times, 50)), 3) if times else 0,
            "p95": round(float(np.percentile(times, 95)), 3) if times else 0,
            "p99": round(float(np.percentile(times, 99)), 3) if times else 0
        },
        "input_stats": {
            "avg_files_per_request": round(np.mean(all_files), 1) if all_files else 0,
            "avg_age": round(np.mean(all_ages), 1) if all_ages else 0,
            "total_requests": len(records),
            "success_rate": f"{success_count / len(records) * 100:.1f}%" if records else "0%"
        }
    }