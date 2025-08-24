import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from infer import model_fn, input_fn, predict_fn

app = FastAPI(title="12-Lead ECG YOLO Inference")

# Load model at startup
MODEL_DIR = "."  # folder containing model.pt
model_dict = model_fn(MODEL_DIR)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Match uploaded filename to a local .npy file and run prediction.
    """
    # Strip extension and build local path
    base_name = os.path.splitext(file.filename)[0]
    npy_file = f"{base_name}.npy"

    if not os.path.exists(npy_file):
        raise HTTPException(status_code=404, detail=f"Local file {npy_file} not found")

    # Load .npy file from disk
    with open(npy_file, "rb") as f:
        content = f.read()

    # Convert content → numpy array using input_fn
    input_data = input_fn(content, "application/x-npy")

    # Run inference
    prediction = predict_fn(input_data, model_dict)

    return prediction
