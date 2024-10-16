import time
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr, final_lr, last_step=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup 階段：從 initial_lr 線性增加到 max_lr (initial_lr)
            return [self.initial_lr * (self.last_epoch / self.warmup_steps) for _ in self.base_lrs]
        else:
            # 餘弦退火階段：從 initial_lr 降低到 final_lr
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.final_lr + (self.initial_lr - self.final_lr) * cosine_decay for _ in self.base_lrs]
            
def save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, eval_loss, eval_accuracy, cfg, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'cfg': cfg
    }
    torch.save(checkpoint, filename)

def train(model, train_loader, optimizer, criterion, device, epoch, log_file, scheduler):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add Elastic Net regularization
        reg_loss = model.get_regularization()
        loss += reg_loss

        loss.backward()
        optimizer.step()
        
        # Update learning rate after each batch
        scheduler.step()
        current_lr = scheduler.get_lr()[0]  # 获取当前学习率
        
        total_loss += loss.item()
        total_samples += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'accuracy': f"{correct_predictions/total_samples:.4f}",
            'lr': f"{current_lr:.6f}"
        })

        # 每100个批次记录一次
        if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            batch_loss = total_loss / (batch_idx + 1)
            batch_accuracy = correct_predictions / total_samples
            log_entry = f"Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} - "
            log_entry += f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}, LR: {current_lr:.6f}\n"
            log_file.write(log_entry)
            log_file.flush()  # 确保立即写入文件

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    epoch_time = time.time() - start_time

    log_entry = f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
    log_entry += f"Time: {epoch_time:.2f}s, Final LR: {current_lr:.6f}\n"
    print(log_entry.strip())
    log_file.write(log_entry)
    log_file.flush()

    return epoch_loss, epoch_accuracy

def evaluate(model, eval_loader, criterion, device, log_file):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluation", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    eval_loss = total_loss / len(eval_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    eval_time = time.time() - start_time

    log_entry = f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {eval_time:.2f}s\n"
    print(log_entry.strip())
    log_file.write(log_entry)

    return eval_loss, accuracy