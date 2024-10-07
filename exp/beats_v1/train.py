import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
import random
import numpy as np
from src.beats_adapt import BEATsForAudioClassification
from src.utils.AudioDataset import AudioDataset4raw, split_annotations
import src.training_process as TP
import psutil
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Classification Training")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (default: 8)')
    parser.add_argument('--initial_lr', type=float, default=5e-5, help='Initial learning rate for Cosine Annealing (default: 5e-5)')
    parser.add_argument('--final_lr', type=float, default=1e-7, help='Final learning rate for Cosine Annealing (default: 1e-7)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--model_path', type=str, default='BEATs_iter3_plus_AS20K.pt', help='Pre-trained model path (default: BEATs_iter3_plus_AS20K.pt)')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of classification labels (default: 3)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--l1_reg', type=float, default=0.0001, help='L1 regularization (default: 0.0001)')
    parser.add_argument('--l2_reg', type=float, default=0.0001, help='L2 regularization (default: 0.0001)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights (default: False)')
    parser.add_argument('--use_augmentation', action='store_true', help='Use audio augmentation (default: False)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='', help='Custom output directory name (default: timestamp-based directory)')
    parser.add_argument('--use_cpu', action='store_true', help='Force using CPU even if GPU is available (default: False)')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_system_info(log_file):
    log_file.write("System Information:\n")
    log_file.write(f"Python version: {sys.version}\n")
    log_file.write(f"PyTorch version: {torch.__version__}\n")
    log_file.write(f"CPU Count: {psutil.cpu_count()}\n")
    log_file.write(f"Total RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB\n")
    
    if torch.cuda.is_available():
        log_file.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        log_file.write(f"CUDA version: {torch.version.cuda}\n")
    else:
        log_file.write("No GPU available\n")
    log_file.write("\n")

def log_args(args, log_file):
    log_file.write("Training Arguments:\n")
    for arg, value in vars(args).items():
        log_file.write(f"{arg}: {value}\n")
    log_file.write("\n")

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Open log file
    log_file = open(os.path.join(output_dir, "training_log.txt"), "w")

    # 記錄系統信息
    log_system_info(log_file)
    
    # 記錄訓練參數
    log_args(args, log_file)

    # Split annotations
    split_annotations('ASDor_wav/train_annotations.csv', os.path.join(output_dir, "tr_annotations.csv"), 
        os.path.join(output_dir, "cv_annotations.csv"), train_ratio=0.82, random_state=args.seed)

    # Create datasets
    train_dataset = AudioDataset4raw(os.path.join(output_dir, "tr_annotations.csv"), augment=args.use_augmentation)
    valid_dataset = AudioDataset4raw(os.path.join(output_dir, "cv_annotations.csv"), augment=False)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    print("Setting up device...")
    # Set device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print GPU info if using CUDA
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")

    print("Loading model...")
    # 記錄模型架構
    log_file.write("Model Architecture:\n")
    model = BEATsForAudioClassification(num_labels=args.num_labels, dropout_rate=args.dropout_rate, 
        l1_reg=args.l1_reg, l2_reg=args.l2_reg, freeze_encoder=args.freeze_encoder).to(device)
    model.initialize_beats(args.model_path, device)
    model = model.to(device)
    log_file.write(str(model) + "\n\n")
    print(f"Model successfully loaded and moved to {device}")

    # Print model parameter status
    for name, param in model.named_parameters():
        print(f"Parameter {name}: requires_grad = {param.requires_grad}")

    print("Setting up optimizer, loss function and learning rate scheduler...")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.initial_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.final_lr)

    print("Starting training loop...")
    best_model_info = {
        'epoch': 0,
        'eval_accuracy': 0,
        'eval_loss': float('inf')
    }

    log_file.write("Training Progress:\n")
    for epoch in range(args.num_epochs):
        epoch_start_time = datetime.now()

        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_accuracy = TP.train(model, train_loader, optimizer, criterion, device, epoch+1, log_file)
        print("Evaluating...")
        eval_loss, eval_accuracy = TP.evaluate(model, valid_loader, criterion, device, log_file)
        
        scheduler.step()
        
        # Get and print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        print(f"Epoch {epoch+1}/{args.num_epochs} summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

        log_entry = f"Epoch {epoch+1}/{args.num_epochs} - "
        log_entry += f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        log_entry += f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}, "
        log_entry += f"Duration: {epoch_duration:.2f}s, "
        log_entry += f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"

        if torch.cuda.is_available():
            log_entry += f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.memory_reserved() / 1e9:.2f}GB\n"

        log_file.write(log_entry)

        # 保存最佳模型
        if eval_accuracy > best_model_info['eval_accuracy']:
            best_model_info = {
                'epoch': epoch + 1,
                'eval_accuracy': eval_accuracy,
                'eval_loss': eval_loss,
                'train_accuracy': train_accuracy,
                'train_loss': train_loss
            }
            model.save_model(os.path.join(output_dir, "best_model.pth"), optimizer, epoch+1, 
                            train_loss, train_accuracy, eval_loss, eval_accuracy)
            log_file.write(f"New best model saved at epoch {epoch+1}\n")

    # 在訓練結束時，記錄最佳模型信息
    log_file.write("\n" + "="*50 + "\n")
    log_file.write("Training completed. Best model summary:\n")
    log_file.write(f"Best epoch: {best_model_info['epoch']}\n")
    log_file.write(f"Best evaluation accuracy: {best_model_info['eval_accuracy']:.4f}\n")
    log_file.write(f"Best evaluation loss: {best_model_info['eval_loss']:.4f}\n")
    log_file.write(f"Corresponding training accuracy: {best_model_info['train_accuracy']:.4f}\n")
    log_file.write(f"Corresponding training loss: {best_model_info['train_loss']:.4f}\n")

    # 保存最佳模型信息到 JSON 文件
    with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
        json.dump(best_model_info, f, indent=4)

    log_file.close()

if __name__ == "__main__":
    main()