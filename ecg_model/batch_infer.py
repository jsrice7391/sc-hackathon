# batch_infer.py
import os
import glob
import json
import csv
import argparse
import numpy as np
import torch

# Use endpoint-style loaders from your updated infer.py
from infer import model_fn, predict_fn, YOLO12Channel  # YOLO12Channel used for manual load

# Optional metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_model_from_checkpoint(checkpoint_path: str):
    """Load checkpoint (best.pt/model.pt) into the same dict format model_fn returns."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]
    yolo_model = ckpt.get("yolo_model", "yolov8n-cls.pt")

    model = YOLO12Channel(yolo_model, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {"model": model, "class_names": class_names}

def load_model_any(model_path: str):
    """
    Be flexible:
      - If model_path is a directory and contains model.pt -> use model_fn(dir)
      - If model_path is a file (e.g., best.pt) -> manual load
      - If model_path is a dir with some .pt -> pick the first .pt and manual load
    """
    if os.path.isdir(model_path):
        mp = os.path.join(model_path, "model.pt")
        if os.path.exists(mp):
            return model_fn(model_path)
        pt_files = glob.glob(os.path.join(model_path, "*.pt"))
        if pt_files:
            return build_model_from_checkpoint(pt_files[0])
        raise FileNotFoundError(f"No .pt checkpoint found in directory: {model_path}")
    else:
        # file path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        return build_model_from_checkpoint(model_path)

def infer_file(npy_path: str, model_dict: dict):
    """Run predict_fn for a single .npy file path."""
    spec = np.load(npy_path)  # shape (12, freq, time)
    pred = predict_fn(spec, model_dict)
    # Attach file path for downstream reporting
    pred["file"] = npy_path
    return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/classify/yolo_12ch/best.pt", help="Path to model .pt or directory with model.pt")
    parser.add_argument("--data_root", default="dataset/test", help="Root folder containing class dirs (e.g., normal/abnormal)")
    parser.add_argument("--csv_out", default="batch_results.csv", help="CSV output path")
    parser.add_argument("--json_out", default="batch_results.json", help="JSON output path")
    args = parser.parse_args()

    # Load model (works for both endpoint bundle dirs and raw .pt files)
    model_dict = load_model_any(args.model)

    # Collect files
    class_dirs = ["abnormal", "normal"]
    npy_files = []
    for c in class_dirs:
        npy_files.extend(glob.glob(os.path.join(args.data_root, c, "*.npy")))

    print(f"🔍 Looking for test files in {args.data_root}")
    print(f"Found {len(npy_files)} files")
    if not npy_files:
        print("❌ No .npy files found.")
        return

    # Inference loop
    results = []
    errors = 0
    for f in npy_files:
        try:
            pred = infer_file(f, model_dict)
            pred["true_class"] = os.path.basename(os.path.dirname(f))  # from folder name
            results.append(pred)
        except Exception as e:
            errors += 1
            print(f"Error processing {f}: {e}")

    if not results:
        print("❌ No results returned (every file errored).")
        return

    # Save JSON
    with open(args.json_out, "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"💾 Results saved to {args.json_out}")

    # Build CSV
    # Collect full class set from probabilities keys (class names)
    all_classes = sorted({cls for r in results for cls in r["probabilities"].keys()})
    header = ["file", "true_class", "predicted_class", "predicted_name", "confidence"] + all_classes

    with open(args.csv_out, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(header)
        for r in results:
            row = [
                r["file"],
                r["true_class"],
                r["predicted_class"],
                r["predicted_name"],
                r["confidence"],
            ] + [r["probabilities"].get(cls, 0.0) for cls in all_classes]
            writer.writerow(row)

    print(f"💾 Results saved to {args.csv_out}")
    if errors:
        print(f"⚠️ {errors} file(s) failed.")

    # -------- Metrics --------
    y_true = [r["true_class"] for r in results]
    y_pred = [r["predicted_name"] for r in results]

    print("\n📊 Classification Metrics:")
    print(classification_report(y_true, y_pred, digits=4))

    labels_for_cm = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
    print("🔀 Confusion Matrix (rows = true, cols = pred):")
    print("Classes:", labels_for_cm)
    print(cm)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Overall Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
