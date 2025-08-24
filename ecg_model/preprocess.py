import os
import json
import numpy as np
from scipy.signal import spectrogram
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_ecg_data(json_path):
    """Extract 12-lead ECG data from your JSON structure."""
    print(f"Processing: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Debug: print the structure
    print(f"JSON keys: {list(data.keys())}")
    
    # Get filename without extension as the key
    filename = Path(json_path).stem.replace('_digitized', '')
    print(f"Looking for key: {filename}")
    
    # Navigate: root -> filename -> leads
    if filename in data:
        ecg_data = data[filename]
        print(f"Found data for {filename}, keys: {list(ecg_data.keys())}")
    else:
        print(f"Available keys in JSON: {list(data.keys())}")
        # Try the first key if filename doesn't match
        first_key = list(data.keys())[0]
        print(f"Using first key: {first_key}")
        ecg_data = data[first_key]
    
    # Extract 12 leads
    leads = []
    for i in range(1, 13):
        lead_key = f'lead_{i}'
        if lead_key not in ecg_data:
            print(f"Available lead keys: {[k for k in ecg_data.keys() if 'lead' in k.lower()]}")
            raise ValueError(f"Missing {lead_key} in {json_path}")
        leads.append(np.array(ecg_data[lead_key]))
        print(f"  {lead_key}: {len(ecg_data[lead_key])} samples")
    
    return np.array(leads)  # Shape: (12, signal_length)

def create_spectrogram(ecg_leads, fs=200):
    """Convert 12-lead ECG to spectrogram tensor."""
    spectrograms = []
    
    for lead in ecg_leads:
        f, t, Sxx = spectrogram(lead, fs=fs, nperseg=256, noverlap=128)
        # Convert to log-power (dB)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        spectrograms.append(Sxx_db)
    
    return np.stack(spectrograms, axis=0)  # (12, freq_bins, time_bins)

def process_dataset(input_dir, output_dir, test_size=0.2, val_size=0.2):
    """
    Process ECG JSON files into YOLO dataset format.
    
    input_dir structure:
    input_dir/
      class_a/
        file1_digitized.json
        file2_digitized.json
      class_b/
        file1_digitized.json
        file2_digitized.json
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Process each class
    all_files = []
    all_labels = []
    class_names = []
    
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        class_names.append(class_name)
        class_idx = len(class_names) - 1
        
        json_files = list(class_dir.glob('*.json'))
        print(f"Found {len(json_files)} JSON files in class '{class_name}'")
        
        for json_file in tqdm(json_files, desc=f"Processing {class_name}"):
            try:
                # Extract ECG data
                ecg_leads = extract_ecg_data(json_file)
                
                # Create spectrogram
                spec = create_spectrogram(ecg_leads)
                
                # Save as NPY
                npy_filename = f"{json_file.stem}.npy"
                all_files.append((spec, npy_filename, class_idx))
                all_labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    # Split dataset
    files_array = np.array(all_files, dtype=object)
    
    # First split: train+val vs test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        files_array, all_labels, test_size=test_size, stratify=all_labels, random_state=42
    )
    
    # Second split: train vs val
    if val_size > 0:
        train_files, val_files, _, _ = train_test_split(
            train_val_files, train_val_labels, 
            test_size=val_size/(1-test_size), 
            stratify=train_val_labels, 
            random_state=42
        )
    else:
        train_files = train_val_files
        val_files = []
    
    # Save files
    splits = [('train', train_files)]
    if len(val_files) > 0:
        splits.append(('val', val_files))
    if len(test_files) > 0:
        splits.append(('test', test_files))
    
    for split_name, split_files in splits:
        for spec, filename, class_idx in tqdm(split_files, desc=f"Saving {split_name}"):
            class_name = class_names[class_idx]
            
            # Create directory
            split_dir = output_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Save spectrogram
            save_path = split_dir / filename
            np.save(save_path, spec)
    
    # Create dataset.yaml
    yaml_content = f"""path: {output_path.absolute()}
train: train
val: val
names:
"""
    for idx, class_name in enumerate(class_names):
        yaml_content += f"  {idx}: {class_name}\n"
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset created:")
    print(f"  Classes: {class_names}")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val: {len(val_files)} samples") 
    print(f"  Test: {len(test_files)} samples")
    print(f"  YAML: {yaml_path}")
    
    return str(yaml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input directory (e.g., ecg_data/data)')
    parser.add_argument('--output', required=True, help='Output dataset directory')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Debug: check input directory
    input_path = Path(args.input)
    print(f"Input directory: {input_path}")
    print(f"Directory exists: {input_path.exists()}")
    if input_path.exists():
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        print(f"Subdirectories: {[d.name for d in subdirs]}")
        for subdir in subdirs:
            json_count = len(list(subdir.glob('*.json')))
            print(f"  {subdir.name}: {json_count} JSON files")
    
    yaml_path = process_dataset(args.input, args.output, args.test_size, args.val_size)
    print(f"\nUse this for training: {yaml_path}")