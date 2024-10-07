import time
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score

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

def train(model, train_loader, optimizer, criterion, device, epoch, log_file):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add Elastic Net regularization
        reg_loss = model.get_regularization()
        loss += reg_loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'accuracy': f"{correct_predictions/total_samples:.4f}"
        })

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    epoch_time = time.time() - start_time

    log_entry = f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Time: {epoch_time:.2f}s\n"
    print(log_entry.strip())
    log_file.write(log_entry)

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