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
    parser = argparse.ArgumentParser(description="Audio Contrastive Learning Testing")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to the test CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--n_enrollment', type=int, default=5,
                        help='Number of enrollment samples per class')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU for testing')
    parser.add_argument('--prefix', type=str, default='None',
                        help='prefix of results file name')
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

def select_enrollment_samples(features, labels, files, n_per_class):
    """為每個類別選擇註冊樣本"""
    unique_labels = torch.unique(labels)
    enrollment_indices = []
    test_indices = []
    
    for label in unique_labels:
        class_indices = (labels == label).nonzero().view(-1)
        
        if len(class_indices) <= n_per_class:
            enroll_idx = class_indices
        else:
            perm = torch.randperm(len(class_indices))
            enroll_idx = class_indices[perm[:n_per_class]]
        
        test_idx = torch.tensor([i for i in class_indices if i not in enroll_idx])
        
        enrollment_indices.extend(enroll_idx.tolist())
        test_indices.extend(test_idx.tolist())
    
    return enrollment_indices, test_indices

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

def normalize_scores(raw_scores, method='snorm', enrollment_scores=None):
    """正規化分數，增加了錯誤處理
    
    Parameters:
        raw_scores: 原始分數向量
        method: 'znorm', 'tnorm', 或 'snorm'
        enrollment_scores: 用於計算正規化參數的註冊分數
        
    Returns:
        normalized_scores: 正規化後的分數
    """
    if method == 'raw':
        return raw_scores
    
    # 確保輸入是有效的
    raw_scores = torch.as_tensor(raw_scores)
    if enrollment_scores is not None:
        enrollment_scores = torch.as_tensor(enrollment_scores)
    
    # 初始化結果
    z_scores = raw_scores
    t_scores = raw_scores
    
    if method in ['znorm', 'snorm'] and enrollment_scores is not None:
        # 確保有足夠的樣本來計算統計量
        if len(enrollment_scores) > 1:
            z_mean = enrollment_scores.mean()
            z_std = enrollment_scores.std()
            # 避免除以零，使用較小的閾值
            if z_std > 1e-8:
                z_scores = (raw_scores - z_mean) / z_std
            else:
                z_scores = raw_scores - z_mean
    
    if method in ['tnorm', 'snorm']:
        # 確保有足夠的分數來計算統計量
        if len(raw_scores) > 1:
            t_mean = raw_scores.mean()
            t_std = raw_scores.std()
            # 避免除以零
            if t_std > 1e-8:
                t_scores = (raw_scores - t_mean) / t_std
            else:
                t_scores = raw_scores - t_mean
    
    if method == 'znorm':
        return z_scores
    elif method == 'tnorm':
        return t_scores
    else:  # snorm
        return 0.5 * (z_scores + t_scores)

def evaluate_classification(features, labels, center_features, enrollment_indices, test_indices):
    """使用原始餘弦相似度進行分類評估"""
    correct = 0
    total = len(test_indices)
    predictions = []
    ground_truths = []
    all_scores = []
    
    for idx in test_indices:
        test_feature = features[idx]
        true_label = labels[idx].item()
        
        # 計算與所有類別中心的相似度
        class_scores = {}
        
        for label, center in center_features.items():
            # 計算與中心的相似度
            similarity = torch.nn.functional.cosine_similarity(
                test_feature.unsqueeze(0),
                center.unsqueeze(0)
            ).item()
            
            class_scores[label] = {
                'raw': similarity,
                'score': similarity  # 使用原始分數作為最終分數
            }
        
        # 使用原始相似度進行預測
        pred_label = max(class_scores.items(), key=lambda x: x[1]['score'])[0]
        
        if pred_label == true_label:
            correct += 1
        
        predictions.append(pred_label)
        ground_truths.append(true_label)
        all_scores.append(class_scores)
    
    accuracy = correct / total
    return accuracy, predictions, ground_truths, all_scores

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

def analyze_features(features, labels, files, n_per_class, output_dir, prefix='None'):
    """進行特徵分析和分類評估 (只使用原始分數)"""
    # 選擇註冊樣本
    enrollment_indices, test_indices = select_enrollment_samples(features, labels, files, n_per_class)
    
    # 計算類別中心
    center_features = calculate_center_features(features, labels, enrollment_indices)
    
    # 評估分類效果
    accuracy, predictions, ground_truths, all_scores = evaluate_classification(
        features, labels, center_features, enrollment_indices, test_indices
    )
    
    # 計算質量指標
    quality_metrics = calculate_quality_metrics(features, labels)
    
    # 生成分類結果報告
    results = []
    for idx, (pred, true, scores) in enumerate(zip(predictions, ground_truths, all_scores)):
        file = files[test_indices[idx]]
        
        # 獲取最高分數和次高分數
        raw_scores = {k: v['raw'] for k, v in scores.items()}
        sorted_scores = sorted(raw_scores.values(), reverse=True)
        score_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        results.append({
            'file': file,
            'true_label': true,
            'predicted_label': pred,
            'correct': pred == true,
            'max_score': sorted_scores[0],
            'score_margin': score_margin,
            **{f'score_to_class_{k}': v['raw'] for k, v in scores.items()}
        })
    
    # 保存結果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f'{prefix}_classification_results.csv'), index=False)
    
    # 繪製分數分布圖
    correct_scores = df[df['correct']]['max_score'].dropna()
    wrong_scores = df[~df['correct']]['max_score'].dropna()
    
    if len(correct_scores) > 0 or len(wrong_scores) > 0:
        plt.figure(figsize=(10, 6))
        if len(correct_scores) > 0:
            plt.hist(correct_scores, bins=min(30, len(correct_scores)), 
                    alpha=0.5, label='Correct', density=True)
        if len(wrong_scores) > 0:
            plt.hist(wrong_scores, bins=min(30, len(wrong_scores)), 
                    alpha=0.5, label='Wrong', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Distribution of Raw Cosine Similarity Scores')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_score_distribution.png'))
        plt.close()
    
    # 繪製混淆矩陣
    unique_labels = sorted(list(center_features.keys()))
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    for true, pred in zip(ground_truths, predictions):
        true_idx = unique_labels.index(true)
        pred_idx = unique_labels.index(pred)
        confusion_matrix[true_idx][pred_idx] += 1
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, 
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                annot=True, 
                fmt='g')
    plt.title('Raw Cosine Similarity Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'n_enrollment_samples': n_per_class,
        'n_test_samples': len(test_indices),
        'n_classes': len(center_features),
        'quality_ratio': quality_metrics['quality_ratio'],
        'intra_class_sim': quality_metrics['intra_class_sim'],
        'inter_class_sim': quality_metrics['inter_class_sim']
    }

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
    
    results = {}
    
    print(f"\nPerforming {args.prefix} analysis...")
    results = analyze_features(
        features, labels, files,
        n_per_class=args.n_enrollment,
        output_dir=args.output_dir,
        prefix=args.prefix
    )
    
    # 保存總結報告
    with open(os.path.join(args.output_dir, f'{args.prefix}_classification_summary.txt'), 'w') as f:
        f.write("Classification Summary Report\n")
        f.write("=========================\n\n")
        f.write(f"Number of enrollment samples per class: {args.n_enrollment}\n")
        f.write(f"Total number of classes: {results['n_classes']}\n")
        f.write(f"Total number of test samples: {results['n_test_samples']}\n\n")
        f.write("Evaluation Metrics by Method:\n")

        f.write(f"\n{args.prefix}:\n")
        f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"  Quality Ratio: {results['quality_ratio']:.4f}\n")
        f.write(f"  Intra-class Similarity: {results['intra_class_sim']:.4f}\n")
        f.write(f"  Inter-class Similarity: {results['inter_class_sim']:.4f}\n")

    print("\nResults Summary:")
    print(f"\n{args.prefix}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Quality Ratio: {results['quality_ratio']:.4f}")
    print(f"  Intra-class Similarity: {results['intra_class_sim']:.4f}")
    print(f"  Inter-class Similarity: {results['inter_class_sim']:.4f}")

    print(f"\nResults saved in {args.output_dir}")

if __name__ == "__main__":
    main()