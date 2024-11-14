import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('src/unilm/beats')
from BEATs import BEATs, BEATsConfig

class ConvProjection(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批標準化
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = x.mean(dim=2)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class JustMeanProjection(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批標準化
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        # x 的形狀: [batch_size, sequence_length, input_dim]
        # 直接在序列維度上進行平均
        x = x.mean(dim=1)  # [batch_size, input_dim]
        #x = self.projection(x)
        # L2 正規化確保特徵在單位球面上
        x = F.normalize(x, p=2, dim=1)
        return x

class SimpleFewShotProjection(nn.Module):
    def __init__(self, input_dim=768, projection_dim=128, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批標準化
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 簡單的兩層結構,不使用dropout
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),  # 使用LayerNorm替代BatchNorm,對小批量更穩定
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 可學習的縮放因子
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        
        # 1. 加權平均
        weights = F.softmax(torch.sum(x, dim=-1), dim=-1)  # [batch_size, seq_len]
        x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]
        
        # 2. 簡單投影
        x = self.projection(x)
        
        # 3. 縮放和正則化
        x = x * self.scale
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # 計算特徵相似度矩陣
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 創建標籤掩碼
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除對角線元素的影響
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        
        mask = mask * logits_mask
        
        # 使用 log_sum_exp trick 來提高數值穩定性
        max_similarity = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix - max_similarity.detach()) * logits_mask
        log_prob = similarity_matrix - max_similarity.detach() - torch.log(exp_sim.sum(1, keepdim=True) + 1e-7)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        loss = -mean_log_prob_pos.mean()
        
        return loss, similarity_matrix

class BEATsContrastive(nn.Module):
    def __init__(self, projection_dim=128, hidden_dim=512, temperature=0.07, 
                 freeze_encoder=True, l1_reg=0.01, l2_reg=0.01):  # 添加正則化參數
        super().__init__()
        self.cfg = None
        self.beats = None
        self.projection = ConvProjection(768, projection_dim, hidden_dim)
        #self.projection = SimpleFewShotProjection()
        self.contrastive_loss = SupConLoss(temperature)
        self.freeze_encoder = freeze_encoder
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
    def initialize_beats(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        self.cfg = BEATsConfig(checkpoint['cfg'])
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(checkpoint['model'])
        self.beats.to(device)

        if self.freeze_encoder:
            self.freeze_beats_encoder()
    
    def freeze_beats_encoder(self):
        for param in self.beats.parameters():
            param.requires_grad = False

    def unfreeze_beats_encoder(self):
        for param in self.beats.parameters():
            param.requires_grad = True
        
    def extract_features(self, input_features):
        """
        提取特徵（用於訓練和推論）
        """
        if self.beats is None:
            raise RuntimeError("BEATs model is not initialized. Call initialize_beats() first.")
            
        padding_mask = torch.zeros(input_features.shape[0], input_features.shape[1], 
                                dtype=torch.bool, device=input_features.device)
        
        with torch.set_grad_enabled(self.training):
            features = self.beats.extract_features(input_features, padding_mask=padding_mask)[0]
            projected_features = self.projection(features)
            
        return projected_features
        
    def get_regularization_loss(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2)
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss
    
    def forward(self, input_features, labels=None):
        """
        前向傳播：
        - 如果提供 labels，進行對比學習並返回損失
        - 如果沒有 labels，則只提取特徵
        """
        projected_features = self.extract_features(input_features)
        
        if labels is not None:
            contrastive_loss, similarity_matrix = self.contrastive_loss(projected_features, labels)
            reg_loss = self.get_regularization_loss()
            total_loss = contrastive_loss + reg_loss
            return total_loss, projected_features, similarity_matrix
            
        return projected_features

    def save_model(self, save_path, optimizer=None, epoch=None):
        """
        保存模型狀態
        Args:
            save_path: 保存路徑
            optimizer: 優化器（可選）
            epoch: 當前訓練輪次（可選）
        """
        if self.beats is None:
            raise RuntimeError("BEATs model is not initialized. Call initialize_beats() first.")

        checkpoint = {
            'model_config': {
                'projection_dim': self.projection.projection[-1].out_features,
                'hidden_dim': self.projection.conv1.out_channels,
                'temperature': self.contrastive_loss.temperature
            },
            'beats_config': self.cfg,
            'beats_state_dict': self.beats.state_dict(),
            'projection_state_dict': self.projection.state_dict(),
            'epoch': epoch
        }

        # 保存優化器狀態（如果提供）
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # 保存模型
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path, device):
        """
        載入模型狀態
        Args:
            load_path: 模型檔案路徑
            device: 執行裝置
        Returns:
            model: 載入的模型
            checkpoint: 完整的檢查點數據
        """
        # 載入檢查點
        checkpoint = torch.load(load_path, map_location=device)
        
        # 從保存的配置創建模型
        model = cls(
            projection_dim=checkpoint['model_config']['projection_dim'],
            hidden_dim=checkpoint['model_config']['hidden_dim'],
            temperature=checkpoint['model_config']['temperature']
        )
        
        # 初始化並載入 BEATs
        model.cfg = checkpoint['beats_config']
        model.beats = BEATs(model.cfg)
        model.beats.load_state_dict(checkpoint['beats_state_dict'])
        
        # 載入投影層
        model.projection.load_state_dict(checkpoint['projection_state_dict'])
        
        # 將模型移到指定設備
        model = model.to(device)
        
        print(f"Model loaded from {load_path}")
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
            
        return model, checkpoint


