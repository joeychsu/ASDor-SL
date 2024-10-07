import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.beats_adapt import BEATsForAudioClassification
from src.utils.AudioDataset import AudioDataset4raw

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Classification Testing")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of classification labels')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU for testing')
    return parser.parse_args()

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds

def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

def main():
    args = parse_args()
    
    print("Setting up device...")
    # Set device
    if args.use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = BEATsForAudioClassification(num_labels=args.num_labels).to(device)
    model, checkpoint = model.load_model(args.model_path, device)
    
    # Prepare the test dataset
    test_dataset = AudioDataset4raw(args.test_csv, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Perform testing
    true_labels, pred_labels = test(model, test_loader, device)
    
    # Generate classification report
    class_names = ['ok', 'ng', 'other']
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)
    
    # Save classification report
    with open(f"{args.output_dir}/classification_report.txt", 'w') as f:
        f.write(report)
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, class_names, args.output_dir)
    
    print(f"Test results saved in {args.output_dir}")

if __name__ == "__main__":
    main()