import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

class ECGDataset(Dataset):
    """Simple ECG Dataset class."""
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        
        # Load class names
        with open(self.data_dir / 'dataset.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.class_names = config['names']
        
        # Collect files
        self.samples = []
        split_dir = self.data_dir / split
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_idx = next(k for k, v in self.class_names.items() if v == class_dir.name)
                for npy_file in class_dir.glob('*.npy'):
                    self.samples.append((npy_file, class_idx))
        
        print(f"{split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Load spectrogram: (12, freq, time)
        spec = np.load(file_path)
        
        # Normalize to [0,1]
        spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        
        return torch.FloatTensor(spec_norm), label

class ChannelConverter(nn.Module):
    """Convert 12-channel input to 3-channel for YOLO."""
    
    def __init__(self):
        super().__init__()
        # Simple 1x1 conv to map 12 channels to 3 channels
        self.conv = nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Initialize to reasonable values
        with torch.no_grad():
            # Initialize weights to average the 12 channels into 3
            self.conv.weight.data = torch.ones(3, 12, 1, 1) / 12.0
            self.conv.bias.data.zero_()
    
    def forward(self, x):
        return self.conv(x)

class YOLO12Channel(nn.Module):
    """YOLO model with 12-channel input converter."""
    
    def __init__(self, yolo_model_path='yolov8n-cls.pt', num_classes=2):
        super().__init__()
        
        # Channel converter: 12 -> 3 channels
        self.channel_converter = ChannelConverter()
        
        # Load YOLO model and extract just the neural network
        yolo_wrapper = YOLO(yolo_model_path, task='classify')
        self.yolo_model = yolo_wrapper.model  # Get the actual PyTorch model
        
        # Modify YOLO's final layer for correct number of classes
        self.modify_yolo_head(num_classes)
        
    def modify_yolo_head(self, num_classes):
        """Modify YOLO's classification head for our number of classes."""
        # Find and modify the classification head
        modified = False
        for name, module in self.yolo_model.named_modules():
            if hasattr(module, 'linear') and isinstance(module.linear, nn.Linear):
                in_features = module.linear.in_features
                module.linear = nn.Linear(in_features, num_classes)
                print(f"Modified YOLO head: {in_features} -> {num_classes} classes")
                modified = True
                break
        
        if not modified:
            print("Warning: Could not find classification head to modify")
    
    def forward(self, x):
        # Convert 12 channels to 3 channels
        x = self.channel_converter(x)
        
        # Pass through YOLO model (not the wrapper)
        output = self.yolo_model(x)
        
        # Handle case where YOLO returns tuple or list
        if isinstance(output, (tuple, list)):
            # Return the first element (usually the logits)
            return output[0]
        else:
            return output

def get_num_classes(yaml_path):
    """Get number of classes from dataset yaml."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return len(config['names'])

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc='Validating'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def train_yolo_12ch(data_yaml, yolo_model='yolov8n-cls.pt', epochs=50, batch_size=32, lr=0.001, device='auto'):
    """Train YOLO with 12-channel converter."""
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Get dataset info
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    num_classes = len(config['names'])
    class_names = config['names']
    
    print(f"Dataset: {dataset_path}")
    print(f"Classes ({num_classes}): {list(class_names.values())}")
    
    # Create datasets
    train_dataset = ECGDataset(dataset_path, split='train')
    val_dataset = ECGDataset(dataset_path, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = YOLO12Channel(yolo_model, num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    converter_params = sum(p.numel() for p in model.channel_converter.parameters())
    yolo_params = sum(p.numel() for p in model.yolo_model.parameters())
    
    print(f"Model parameters:")
    print(f"  Channel converter: {converter_params:,}")
    print(f"  YOLO backbone: {yolo_params:,}")
    print(f"  Total: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_acc = 0
    patience = 10
    patience_counter = 0
    
    # Create output directory
    output_dir = Path('runs/classify/yolo_12ch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save complete model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'class_names': class_names,
                'yolo_model': yolo_model,
            }, output_dir / 'best.pt')
            
            print(f"✓ New best model saved! Accuracy: {best_acc:.2f}%")
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    print(f"\n🎉 Training completed!")
    print(f"✓ Best validation accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to: {output_dir / 'best.pt'}")
    
    return model, output_dir / 'best.pt'

def test_model_forward_pass(model, sample_shape=(1, 12, 128, 129)):
    """Test that the model can handle 12-channel input."""
    print(f"\nTesting model with input shape: {sample_shape}")
    
    model.eval()
    dummy_input = torch.randn(sample_shape)
    
    with torch.no_grad():
        try:
            # Test channel converter
            converted = model.channel_converter(dummy_input)
            print(f"✓ Channel converter output: {converted.shape}")
            
            # Test full model
            output = model(dummy_input)
            # print(f"✓ Model output shape: {output.shape}")
            print(f"✓ Output type: {type(output)}")
            
            # Test softmax probabilities
            probs = torch.softmax(output, dim=1)
            print(f"✓ Probabilities sum: {probs.sum(dim=1)}")
            print(f"✓ Sample probabilities: {probs[0]}")
            
            return True
            
        except Exception as e:
            print(f"✗ Model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='dataset.yaml path')
    parser.add_argument('--model', default='yolov8n-cls.pt', help='YOLO model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data).exists():
        print(f"Error: Dataset YAML not found: {args.data}")
        exit(1)
    
    print(f"Training configuration:")
    print(f"  Data: {args.data}")
    print(f"  YOLO model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    try:
        # Get number of classes
        num_classes = get_num_classes(args.data)
        
        # Test model creation and forward pass
        print(f"\nTesting model architecture...")
        test_model = YOLO12Channel(args.model, num_classes)
        
        if test_model_forward_pass(test_model):
            print("✓ Model architecture test passed!")
            
            # Train the model
            model, model_path = train_yolo_12ch(
                args.data,
                args.model, 
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                device=args.device
            )
            
            print(f"\n🎉 Training completed successfully!")
            print(f"✓ Use this model for inference: {model_path}")
            
        else:
            print("✗ Model architecture test failed!")
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()