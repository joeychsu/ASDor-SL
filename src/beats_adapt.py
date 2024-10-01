import torch
from transformers import WhisperModel, WhisperConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torchaudio
import torchaudio.transforms as transforms
import os
import numpy as np
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm
import json
import datetime
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torchaudio.transforms as transforms
import sys
sys.path.append('src/unilm/beats')
from BEATs import BEATs, BEATsConfig
import types

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, sequence_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.mean(dim=2)  # 全局平均池化

# 自訂分類模型
class BEATsForAudioClassification(nn.Module):
    def __init__(self, model_name='BEATs_iter3_plus_AS20K.pt', num_labels=3, dropout_rate=0.1, l1_reg=0.01, l2_reg=0.01, freeze_encoder=True):
        super().__init__()
        # 加載預訓練的BEATs模型
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(checkpoint['model'])
        
        self.tcn = TemporalConvNet(768, hidden_dim=256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, num_labels)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        if freeze_encoder:
            self.freeze_beats_encoder()

    def forward(self, input_features):
        # input_features 的形狀應該是 [batch_size, sequence_length]
        padding_mask = torch.zeros(input_features.shape[0], input_features.shape[1], dtype=torch.bool, device=input_features.device)
        
        # 使用 BEATs 模型提取特徵
        features = self.beats.extract_features(input_features, padding_mask=padding_mask)[0]
        
        # 對特徵進行平均池化
        #pooled_output = features.mean(dim=1)
        pooled_output = self.tcn(features)
        
        x = self.dropout1(pooled_output)
        logits = self.fc1(x)
        return logits

    def freeze_beats_encoder(self):
        for param in self.beats.parameters():
            param.requires_grad = False

    def unfreeze_beats_encoder(self):
        for param in self.beats.parameters():
            param.requires_grad = True

    def get_regularization(self):
        l1_loss = 0.0
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:  # 只对未冻结的参数应用正则化
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2)
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss

    def save_checkpoint(self, model, optimizer, epoch, train_loss, train_accuracy, eval_loss, eval_accuracy, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy
        }
        torch.save(checkpoint, filename)

    

    
