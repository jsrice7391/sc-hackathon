# infer.py
import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from scipy.signal import spectrogram
from pathlib import Path

# ========================
# Model definition
# ========================

class ChannelConverter(nn.Module):
    """Convert 12-channel input to 3-channel for YOLO."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0, bias=True)
        with torch.no_grad():
            self.conv.weight.data = torch.ones(3, 12, 1, 1) / 12.0
            self.conv.bias.data.zero_()
    def forward(self, x):
        return self.conv(x)

class YOLO12Channel(nn.Module):
    """YOLO model with 12-channel input converter."""
    def __init__(self, yolo_model_path='yolov8n-cls.pt', num_classes=2):
        super().__init__()
        self.channel_converter = ChannelConverter()
        yolo_wrapper = YOLO(yolo_model_path)
        self.yolo_model = yolo_wrapper.model
        self.modify_yolo_head(num_classes)
    def modify_yolo_head(self, num_classes):
        for name, module in self.yolo_model.named_modules():
            if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear):
                in_features = module.linear.in_features
                module.linear = nn.Linear(in_features, num_classes)
                break
    def forward(self, x):
        x = self.channel_converter(x)
        return self.yolo_model(x)

# ========================
# Preprocessing
# ========================

def extract_ecg_data(json_data, filename=None):
    """Extract 12-lead ECG data from dict loaded from JSON."""
    if filename:
        key = Path(filename).stem.replace('_digitized', '')
        ecg_data = json_data.get(key, json_data[list(json_data.keys())[0]])
    else:
        ecg_data = json_data[list(json_data.keys())[0]]

    leads = []
    for i in range(1, 13):
        lead_key = f'lead_{i}'
        if lead_key not in ecg_data:
            raise ValueError(f"Missing {lead_key}")
        leads.append(np.array(ecg_data[lead_key]))
    return np.array(leads)  # (12, signal_len)

def create_spectrogram(ecg_leads, fs=200):
    """Convert 12-lead ECG to spectrogram tensor."""
    spectrograms = []
    for lead in ecg_leads:
        f, t, Sxx = spectrogram(lead, fs=fs, nperseg=256, noverlap=128)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        spectrograms.append(Sxx_db)
    return np.stack(spectrograms, axis=0)  # (12, freq_bins, time_bins)

# ========================
# SageMaker required functions
# ========================

def model_fn(model_dir):
    """Load model for SageMaker."""
    checkpoint_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    yolo_model = checkpoint.get('yolo_model', 'yolov8n-cls.pt')

    model = YOLO12Channel(yolo_model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return {"model": model, "class_names": class_names}

def input_fn(request_body, request_content_type):
    """Deserialize input request."""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        if "spectrogram" in data:
            # Already preprocessed
            return np.array(data["spectrogram"], dtype=np.float32)
        else:
            # Raw ECG JSON → spectrogram
            ecg_leads = extract_ecg_data(data)
            spec = create_spectrogram(ecg_leads)
            return spec.astype(np.float32)

    elif request_content_type == "application/x-npy":
        buf = io.BytesIO(request_body)
        return np.load(buf)

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Run inference."""
    model = model_dict["model"]
    class_names = model_dict["class_names"]

    spec = input_data
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    tensor = torch.FloatTensor(spec_norm).unsqueeze(0)  # (1,12,freq,time)

    with torch.no_grad():
        outputs = model(tensor)
        while isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=1)[0]

    predicted = torch.argmax(probs).item()
    confidence = probs[predicted].item()

    return {
        "predicted_class": int(predicted),
        "predicted_name": class_names[predicted],
        "confidence": float(confidence),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))}
    }

def output_fn(prediction, response_content_type):
    """Serialize output."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

# ========================
# Local test mode
# ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, default="model.pt", help="Trained model path")
    parser.add_argument("--input", required=True, help="Input file (.json or .npy)")
    args = parser.parse_args()

    model_dict = model_fn(os.path.dirname(args.model))

    if args.input.endswith(".json"):
        with open(args.input, "r") as f:
            data = json.load(f)
        spec = create_spectrogram(extract_ecg_data(data, args.input))
    else:
        spec = np.load(args.input)

    prediction = predict_fn(spec, model_dict)
    print(json.dumps(prediction, indent=2))
