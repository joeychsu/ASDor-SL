import argparse
from datetime import datetime
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.training_process import WarmupCosineSchedule
from tqdm import tqdm
import logging
import random
import numpy as np
from src.beats_cl import BEATsContrastive  # 改成對比學習模型
from src.utils.AudioDataset import AudioDataset4raw
import psutil
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Contrastive Learning Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 自動補上默認值
    )
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--initial_lr', type=float, default=1e-4, 
                        help='Initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-6, 
                        help='Final learning rate')
    parser.add_argument('--l1_reg', type=float, default=0.00,
                        help='L1 regularization weight')
    parser.add_argument('--l2_reg', type=float, default=0.01,
                        help='L2 regularization weight')
    parser.add_argument('--warmup_epochs', type=int, default=10, 
                        help='Number of epochs for warmup')
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--model_path', type=str, default='BEATs_iter3_plus_AS20K.pt', 
                        help='Pre-trained model path')
    parser.add_argument('--projection_dim', type=int, default=128, 
                        help='Dimension of projection space')
    parser.add_argument('--hidden_dim', type=int, default=512, 
                        help='Hidden dimension for projection')
    parser.add_argument('--temperature', type=float, default=0.2, 
                        help='Temperature parameter for contrastive loss')
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help='Freeze encoder weights')
    parser.add_argument('--use_augmentation', action='store_true', 
                        help='Use audio augmentation')
    parser.add_argument('--augment_type', type=str, default='None',
                        help='Specify augmentation type')
    parser.add_argument('--augment_prob', type=float, default=0.5,
                        help='Probability of applying augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--train_csv', type=str, default='results/tr_annotations.csv',
                        help='Path to training CSV file')
    parser.add_argument('--valid_csv', type=str, default='results/cv_annotations.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Custom output directory name (required)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    parser.add_argument('--projection_type', type=str, default='ConvProjection',
                        help='Projection type for the model')
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

def train_epoch(model, train_loader, optimizer, device, scheduler, epoch, log_file):
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    total_reg_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss, projected_features, similarity_matrix = model(data, labels)
        
        # 獲取個別損失值（用於監控）
        with torch.no_grad():
            contrastive_loss, _ = model.contrastive_loss(projected_features, labels)
            reg_loss = model.get_regularization_loss()
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 累積損失
        total_loss += loss.item()
        total_contrastive_loss += contrastive_loss.item()
        total_reg_loss += reg_loss.item()
        
        # 更新進度條
        progress_bar.set_postfix({
            'Total Loss': f'{loss.item():.4f}',
            'Con Loss': f'{contrastive_loss.item():.4f}',
            'Reg Loss': f'{reg_loss.item():.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches
    }

def evaluate(model, valid_loader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(valid_loader, desc='Evaluating'):
            data = data.to(device)
            features = model.extract_features(data)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    # 串接所有特徵和標籤
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # 計算餘弦相似度矩陣
    similarity_matrix = torch.mm(features, features.t())  # 範圍在 [-1, 1]

    # 將相似度值映射到 [0, 1] 範圍：(x + 1) / 2
    similarity_matrix = (similarity_matrix + 1) / 2
    
    # 創建標籤遮罩
    labels = labels.view(-1, 1)
    mask = (labels == labels.T).float()
    
    # 計算類內和類間相似度
    # 注意：這裡值越大表示越相似，與歐氏距離相反
    intra_class_sim = (similarity_matrix * mask).sum() / mask.sum()
    inter_class_sim = (similarity_matrix * (1 - mask)).sum() / (1 - mask).sum()
    
    # 計算質量比率
    # 注意：因為是相似度，所以分子分母關係要顛倒
    # 我們希望類內相似度高(接近1)，類間相似度低(接近0)
    quality_ratio = intra_class_sim / (inter_class_sim + 1e-8)
    
    return {
        'intra_class_sim': intra_class_sim.item(),
        'inter_class_sim': inter_class_sim.item(),
        'quality_ratio': quality_ratio.item()
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 創建輸出目錄
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_contrastive_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 設置日誌文件
    log_file = open(os.path.join(output_dir, "training_log.txt"), "w")
    log_system_info(log_file)
    log_args(args, log_file)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
    
    # 創建數據集
    train_dataset = AudioDataset4raw(args.train_csv, 
                                   augment=args.use_augmentation,
                                   augment_type=args.augment_type,
                                   augment_prob=args.augment_prob,
                                   device=device,
                                   random_cut=True)
    valid_dataset = AudioDataset4raw(args.valid_csv, 
                                   augment=False,
                                   device=device,
                                   random_cut=True)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    
    # 創建模型
    print("Creating model...")
    model = BEATsContrastive(
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        freeze_encoder=args.freeze_encoder,
        l1_reg=args.l1_reg,
        l2_reg=args.l2_reg,
        projection_type=args.projection_type
    ).to(device)
    
    # 初始化BEATs
    print("Initializing BEATs...")
    model.initialize_beats(args.model_path, device)
    log_file.write(str(model) + "\n\n")
    
    # 打印模型參數狀態
    for name, param in model.named_parameters():
        print(f"Parameter {name}: requires_grad = {param.requires_grad}")
    
    # 設置優化器和學習率調度器
    print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.initial_lr)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps, 
                                   args.initial_lr, args.final_lr)
    
    # 追踪最佳模型
    best_model_info = {
        'epoch': 0,
        'quality_ratio': 0,
        'intra_class_sim': float('inf'),
        'inter_class_sim': 0,
        'train_loss': float('inf')
    }
    
    # 訓練循環
    print("Starting training...")
    log_file.write("Training Progress:\n")
    
    for epoch in range(args.num_epochs):
        epoch_start_time = datetime.now()
        
        # 訓練
        train_losses = train_epoch(model, train_loader, optimizer, device, scheduler, epoch+1, log_file)
        
        # 評估
        metrics = evaluate(model, valid_loader, device)
        
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        # 記錄訓練信息
        log_entry = f"Epoch {epoch+1}/{args.num_epochs} - "
        log_entry += f"Total Loss: {train_losses['total_loss']:.4f}, "
        log_entry += f"Contrastive Loss: {train_losses['contrastive_loss']:.4f}, "
        log_entry += f"Reg Loss: {train_losses['reg_loss']:.4f}, "
        log_entry += f"Intra-class Sim: {metrics['intra_class_sim']:.4f}, "
        log_entry += f"Inter-class Sim: {metrics['inter_class_sim']:.4f}, "
        log_entry += f"Quality Ratio: {metrics['quality_ratio']:.4f}, "
        log_entry += f"Duration: {epoch_duration:.2f}s, "
        log_entry += f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n"
        
        if torch.cuda.is_available():
            log_entry += f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB\n"
        
        print(log_entry)
        log_file.write(log_entry)
        
        # 保存最佳模型（仍然基於quality ratio）
        if metrics['quality_ratio'] > best_model_info['quality_ratio']:
            best_model_info = {
                'epoch': epoch + 1,
                'quality_ratio': metrics['quality_ratio'],
                'intra_class_sim': metrics['intra_class_sim'],
                'inter_class_sim': metrics['inter_class_sim'],
                'total_loss': train_losses['total_loss'],
                'contrastive_loss': train_losses['contrastive_loss'],
                'reg_loss': train_losses['reg_loss']
            }
            model.save_model(
                os.path.join(output_dir, "best_model.pth"),
                optimizer=optimizer,
                epoch=epoch+1
            )
            log_file.write(f"New best model saved at epoch {epoch+1}\n")
    
    # 記錄最佳模型信息
    log_file.write("\n" + "="*50 + "\n")
    log_file.write("Training completed. Best model summary:\n")
    for key, value in best_model_info.items():
        log_file.write(f"{key}: {value}\n")
    
    # 保存最佳模型信息到JSON
    with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
        json.dump(best_model_info, f, indent=4)
    
    log_file.close()

if __name__ == "__main__":
    main()