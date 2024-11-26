import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from src.beats_cl import BEATsContrastive
from src.utils.AudioDataset import AudioDataset4raw
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Extracted vector by model")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to the test CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU for testing')
    return parser.parse_args()

def extract_features(model, data_loader, device):
    """提取特徵"""
    model.eval()
    all_features = []
    all_labels = []
    all_files = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(data_loader, desc="Extracting features")):
            inputs = inputs.to(device)
            features = model.extract_features(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels)
            
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + labels.size(0)
            batch_files = data_loader.dataset.audio_labels.iloc[start_idx:end_idx]['filename'].tolist()
            all_files.extend(batch_files)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels, all_files

def calculate_center_features(features, labels, enrollment_indices):
    """計算每個類別的中心特徵"""
    unique_labels = torch.unique(labels)
    center_features = {}
    
    for label in unique_labels:
        class_enroll_indices = [i for i in enrollment_indices if labels[i] == label]
        class_features = features[class_enroll_indices]
        center_feature = torch.mean(class_features, dim=0)
        center_features[label.item()] = center_feature
    
    return center_features

def calculate_quality_metrics(features, labels):
    """計算特徵質量指標，使用餘弦相似度
    
    Args:
        features: 特徵張量 (N x D)
        labels: 標籤張量 (N)
        
    Returns:
        dict: 包含 quality_ratio, intra_class_sim, inter_class_sim 的字典
    """
    # 正規化特徵向量
    features = torch.nn.functional.normalize(features, dim=1)
    
    # 計算餘弦相似度矩陣
    similarity_matrix = torch.mm(features, features.t())  # 範圍在 [-1, 1]

    # 將相似度值映射到 [0, 1] 範圍：(x + 1) / 2
    similarity_matrix = (similarity_matrix + 1) / 2
    
    # 創建標籤匹配矩陣
    labels = labels.view(-1, 1)
    mask = (labels == labels.T).float()
    
    # 計算類內相似度（相同類別的平均相似度）
    intra_class_sim = (similarity_matrix * mask).sum() / (mask.sum() + 1e-8)
    
    # 計算類間相似度（不同類別的平均相似度）
    inter_class_sim = (similarity_matrix * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
    
    # 計算質量比率
    # 因為使用相似度，所以分子分母關係要顛倒
    # 我們希望類內相似度高(接近1)，類間相似度低(接近-1或0)
    quality_ratio = intra_class_sim / (inter_class_sim + 1e-8)
    
    return {
        'quality_ratio': quality_ratio.item(),
        'intra_class_sim': intra_class_sim.item(),
        'inter_class_sim': inter_class_sim.item()
    }

def analyze_features(features, labels, files, output_dir):
    """進行特徵分析和分類評估 (只使用原始分數)"""

    # 計算類別中心
    center_features = calculate_center_features(features, labels, range(len(features))) # 表示拿全部的資料來算 center_features
    
    print(f"Saving enroll vectors into {os.path.join(output_dir, f'enroll_vectors.pt')} ... ")
    torch.save(center_features, os.path.join(output_dir, f'enroll_vectors.pt'))
    
    # 計算質量指標
    quality_metrics = calculate_quality_metrics(features, labels)

    return quality_metrics
    
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"Using device: {device}")
    
    # 載入模型
    print("Loading model...")
    model, checkpoint = BEATsContrastive.load_model(args.model_path, device)
    model.eval()
    
    # 準備數據
    test_dataset = AudioDataset4raw(args.test_csv, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 提取特徵
    print("Extracting features...")
    features, labels, files = extract_features(model, test_loader, device)
    
    # 傳遞隨機種子進行分析
    print(f"\nPerforming analysis...")
    quality_metrics = analyze_features(
        features, labels, files,
        output_dir=args.output_dir
    )
    


if __name__ == "__main__":
    main()