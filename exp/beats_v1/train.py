import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from src.beats_adapt import BEATsForAudioClassification
from src.utils.AudioDataset import AudioDataset4raw, split_annotations
import src.training_process as TP

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Classification Training")
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for Cosine Annealing')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--model_name', type=str, default='BEATs_iter3_plus_AS20K.pt', help='Pre-trained model name')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of classification labels')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--l1_reg', type=float, default=0.0001, help='L1 regularization')
    parser.add_argument('--l2_reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    return parser.parse_args()

def main():
    args = parse_args()
    # 開始執行主函數...
    print("Starting main function...")

    # 建立輸出目錄
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 開啟日誌檔案
    log_file = open(os.path.join(output_dir, "training_log.txt"), "w")

    # 使用函數
    split_annotations('ASDor_wav/train_annotations.csv', os.path.join(output_dir, "tr_annotations.csv"), 
        os.path.join(output_dir, "cv_annotations.csv"), train_ratio=0.82, random_state=42)

    # 訓練集
    train_dataset = AudioDataset4raw(os.path.join(output_dir, "tr_annotations.csv"), augment=True)
    valid_dataset = AudioDataset4raw(os.path.join(output_dir, "cv_annotations.csv"), augment=False)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")

    # 建立資料載入器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 如果使用 GPU，顯示 GPU 資訊
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")

    print("Loading model...")
    model = BEATsForAudioClassification(model_name=args.model_name, num_labels=args.num_labels, 
        dropout_rate=args.dropout_rate, l1_reg=args.l1_reg, l2_reg=args.l2_reg, 
        freeze_encoder=args.freeze_encoder).to(device)
    print(f"Model successfully loaded and moved to {device}")
    log_file.write(f"Pre-trained model : {args.model_name}\n")

    # 印出模型參數狀態
    for name, param in model.named_parameters():
        print(f"Parameter {name}: requires_grad = {param.requires_grad}")

    print("Setting up optimizer, loss function and learning rate scheduler...")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)

    print("Starting training loop...")
    best_eval_accuracy = 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss, train_accuracy = TP.train(model, train_loader, optimizer, criterion, device, epoch+1, log_file)
        print("Evaluating...")
        eval_loss, eval_accuracy = TP.evaluate(model, valid_loader, criterion, device, log_file)
        
        # 調用學習率調度器
        #scheduler.step(eval_loss)
        scheduler.step()
        
        # 獲取並印出當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        print(f"Epoch {epoch+1}/{args.num_epochs} summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")

        # 如果使用 GPU，顯示記憶體使用情況
        if device.type == 'cuda':
            print(f"GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")

        # 保存最佳模型
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            TP.save_checkpoint(model, optimizer, epoch+1, train_loss, train_accuracy, eval_loss, eval_accuracy, 
                            os.path.join(output_dir, "best_model.pth"))

    print("Training completed. Saving final model...")
    TP.save_checkpoint(model, optimizer, args.num_epochs, train_loss, train_accuracy, eval_loss, eval_accuracy, 
                    os.path.join(output_dir, "final_model.pth"))
    print("Model saved successfully.")

    log_file.close()

if __name__ == "__main__":
    main()