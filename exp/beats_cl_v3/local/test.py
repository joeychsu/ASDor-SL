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
    parser.add_argument('--class_vector_pt', type=str, required=True,
                        help='class vector for enroll')
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
    parser.add_argument('--seed', type=int, default=905,
                        help='Random seed for reproducibility')
    parser.add_argument('--ng_scale', type=float, default=1.0,
                        help='Scaling factor for adjusting NG (No Good) sample weights. '
                             'Recommended range is 1.0 to 2.0.')
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

def select_enrollment_samples(features, labels, files, n_per_class, random_state, target_labels):
    """
    為指定的類別選擇註冊樣本，其他剩餘樣本都放入 test_indices
    - features: 特徵向量
    - labels: 標籤 (str 或 int)
    - files: 檔案名稱
    - n_per_class: 每個類別的註冊樣本數
    - random_state: 隨機種子
    - target_labels: 要處理的類別列表 (列表形式)
    """
    enrollment_indices = []
    all_indices = set(range(len(labels)))  # 所有樣本索引
    rng = torch.Generator()
    rng.manual_seed(random_state)

    for target_label in target_labels:
        # 找到符合該類別的所有索引
        class_indices = (labels == target_label).nonzero().view(-1)
        
        if len(class_indices) == 0:
            print(f"Warning: No samples found for label {target_label}")
            continue
        
        # 隨機抽取 n_per_class 作為 enrollment
        if len(class_indices) <= n_per_class:
            enroll_idx = class_indices
        else:
            perm = torch.randperm(len(class_indices), generator=rng)
            enroll_idx = class_indices[perm[:n_per_class]]
        
        enrollment_indices.extend(enroll_idx.tolist())

    # 剩餘的索引作為 test_indices
    enrollment_set = set(enrollment_indices)
    test_indices = list(all_indices - enrollment_set)

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

def evaluate_classification(features, labels, center_features, enrollment_indices, test_indices, class_weights):
    """
    使用原始餘弦相似度進行分類評估，並支持不同類別權重。
    
    Args:
        features: 測試特徵。
        labels: 測試樣本的真實標籤。
        center_features: 每個類別的中心特徵。
        enrollment_indices: 訓練集中使用的樣本索引。
        test_indices: 測試集中使用的樣本索引。
        class_weights: 字典，為每個類別設置的權重，例如 {0: 1.0, 1: 1.5, 2: 2.0}。
        
    Returns:
        accuracy: 總準確率。
        predictions: 預測標籤列表。
        ground_truths: 真實標籤列表。
        all_scores: 所有樣本的分數詳細信息。
        class_accuracy: 每個類別的準確率。
    """

    """將權重正規化，使總和為 1"""
    total_weight = sum(class_weights.values())
    class_weights = {label: weight / total_weight for label, weight in class_weights.items()}

    correct = 0
    total = len(test_indices)
    predictions = []
    ground_truths = []
    all_scores = []
    
    # 統計每個類別的正確數量與總數
    class_correct = {}
    class_total = {}

    for idx in test_indices:
        test_feature = features[idx]
        true_label = labels[idx].item()
        
        # 初始化類別的統計字典
        if true_label not in class_correct:
            class_correct[true_label] = 0
            class_total[true_label] = 0
        
        # 計算與所有類別中心的相似度
        class_scores = {}
        for label, center in center_features.items():
            similarity = torch.nn.functional.cosine_similarity(
                test_feature.unsqueeze(0),
                center.unsqueeze(0)
            ).item()
            scaled_similarity = (similarity + 1) / 2  # 將相似度範圍轉換為 0~1
            
            # 應用類別權重
            weighted_score = scaled_similarity * class_weights.get(label, 1.0)  # 默認權重為 1.0
            
            class_scores[label] = {
                'raw': similarity,
                'scaled': scaled_similarity,
                'weighted': weighted_score
            }
        
        # 使用加權分數進行預測
        pred_label = max(class_scores.items(), key=lambda x: x[1]['weighted'])[0]
        
        # 更新總數和正確數
        class_total[true_label] += 1
        if pred_label == true_label:
            correct += 1
            class_correct[true_label] += 1
        
        predictions.append(pred_label)
        ground_truths.append(true_label)
        all_scores.append(class_scores)
    
    # 計算每個類別的準確率
    class_accuracy = {label: class_correct[label] / class_total[label] 
                      for label in class_total}
    
    accuracy = correct / total
    return accuracy, predictions, ground_truths, all_scores, class_accuracy

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

def mean_current_center_features(data1, data2):
    # 初始化結果字典
    average_result = {}

    # 對每個 key 分別計算平均值
    for key in data1.keys():
        tensor_list = []
        
        # 添加有效的張量（沒有 NaN）
        if not torch.isnan(data1[key]).any():
            tensor_list.append(data1[key])
        if not torch.isnan(data2[key]).any():
            tensor_list.append(data2[key])
        
        # 如果有有效的張量，計算平均值
        if tensor_list:
            average_result[key] = torch.mean(torch.stack(tensor_list), dim=0)
        else:
            average_result[key] = None  # 如果全是 NaN，可以選擇用 None 或其他值標示

    # 輸出結果
    for key, value in average_result.items():
        if value is not None:
            print(f"Key {key}: Average computed, shape {value.shape}")
        else:
            print(f"Key {key}: No valid data, result is None")
    return average_result

def analyze_features(features, labels, files, n_per_class, class_vector_pt_path, output_dir, random_state, ng_scale, prefix='None'):
    """進行特徵分析和分類評估 (只使用原始分數)"""
    # 選擇註冊樣本
    target_labels = [0] # {"ok": 0, "ng": 1, "other": 2} can found on src/utils/AudioDataset.py
    enrollment_indices, test_indices = select_enrollment_samples(features, labels, files, n_per_class, random_state, target_labels)
    
    # 計算類別中心
    current_center_features = calculate_center_features(features, labels, enrollment_indices)
    center_features = torch.load(class_vector_pt_path)
    center_features = mean_current_center_features(current_center_features, center_features)
    
    # 評估分類效果
    accuracy, predictions, ground_truths, all_scores, class_accuracy = evaluate_classification(
        features, labels, center_features, enrollment_indices, test_indices, {0: 1.0, 1: ng_scale, 2: 1.0}
    )

    # 計算質量指標
    quality_metrics = calculate_quality_metrics(features, labels)
    
    # 保存類別準確率報告
    with open(os.path.join(output_dir, f'{prefix}_class_accuracy.txt'), 'w') as f:
        f.write("Class-wise Accuracy Report\n")
        f.write("===========================\n")
        for label, acc in class_accuracy.items():
            f.write(f"Class {label}: Accuracy = {acc:.4f}\n")
    
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
        'class_accuracy': class_accuracy,  # 返回每類別準確率
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
    
    # 傳遞隨機種子進行分析
    print(f"\nPerforming {args.prefix} analysis...")
    results = analyze_features(
        features, labels, files,
        n_per_class=args.n_enrollment,
        class_vector_pt_path=args.class_vector_pt,
        output_dir=args.output_dir,
        random_state=args.seed, 
        ng_scale=args.ng_scale, 
        prefix=args.prefix
    )
    
    # 保存總結報告
    with open(os.path.join(args.output_dir, f'{args.prefix}_classification_summary.txt'), 'w') as f:
        f.write("Classification Summary Report\n")
        f.write("=========================\n\n")
        f.write(f"Number of enrollment samples per class: {args.n_enrollment}\n")
        f.write(f"Total number of classes: {results['n_classes']}\n")
        f.write(f"Total number of test samples: {results['n_test_samples']}\n\n")
        f.write("Evaluation Metrics:\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Quality Ratio: {results['quality_ratio']:.4f}\n")
        f.write(f"Intra-class Similarity: {results['intra_class_sim']:.4f}\n")
        f.write(f"Inter-class Similarity: {results['inter_class_sim']:.4f}\n\n")
        f.write("Class-wise Accuracy:\n")
        for label, acc in results['class_accuracy'].items():
            f.write(f"  Class {label}: Accuracy = {acc:.4f}\n")
        f.write("\n\n")

    print("\nResults Summary:")
    print(f"\n{args.prefix}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Quality Ratio: {results['quality_ratio']:.4f}")
    print(f"  Intra-class Similarity: {results['intra_class_sim']:.4f}")
    print(f"  Inter-class Similarity: {results['inter_class_sim']:.4f}")
    print(f"  Class-wise Accuracy:")
    for label, acc in results['class_accuracy'].items():
        print(f"   - Class {label}: Accuracy = {acc:.4f}")

    print(f"\nResults saved in {args.output_dir}")

if __name__ == "__main__":
    main()