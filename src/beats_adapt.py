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
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.mean(dim=2)

class BEATsForAudioClassification(nn.Module):
    def __init__(self, num_labels=3, dropout_rate=0.1, l1_reg=0.01, l2_reg=0.01, freeze_encoder=True):
        super().__init__()
        self.cfg = None
        self.beats = None
        self.tcn = TemporalConvNet(768, hidden_dim=512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, num_labels)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.freeze_encoder = freeze_encoder

    def initialize_beats(self, model_path, device):
        checkpoint = torch.load(model_path, map_location=device)
        self.cfg = BEATsConfig(checkpoint['cfg'])
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(checkpoint['model'])
        self.beats.to(device)
        
        if self.freeze_encoder:
            self.freeze_beats_encoder()

    def forward(self, input_features):
        if self.beats is None:
            raise RuntimeError("BEATs model is not initialized. Call initialize_beats() first.")
        
        padding_mask = torch.zeros(input_features.shape[0], input_features.shape[1], dtype=torch.bool, device=input_features.device)
        features = self.beats.extract_features(input_features, padding_mask=padding_mask)[0]
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
            if param.requires_grad:
                l1_loss += torch.norm(param, 1)
                l2_loss += torch.norm(param, 2)
        return self.l1_reg * l1_loss + self.l2_reg * l2_loss

    def save_model(self, save_path, optimizer=None, epoch=None, train_loss=None, train_accuracy=None, eval_loss=None, eval_accuracy=None):
        """
        Save the complete model state, including BEATs and additional layers.
        """
        if self.beats is None:
            raise RuntimeError("BEATs model is not initialized. Call initialize_beats() first.")

        checkpoint = {
            'beats_config': self.cfg,
            'beats_state_dict': self.beats.state_dict(),
            'tcn_state_dict': self.tcn.state_dict(),
            'fc1_state_dict': self.fc1.state_dict(),
            'dropout1_state_dict': self.dropout1.state_dict(),
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'freeze_encoder': self.freeze_encoder
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if train_loss is not None:
            checkpoint['train_loss'] = train_loss
        if train_accuracy is not None:
            checkpoint['train_accuracy'] = train_accuracy
        if eval_loss is not None:
            checkpoint['eval_loss'] = eval_loss
        if eval_accuracy is not None:
            checkpoint['eval_accuracy'] = eval_accuracy

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path, device):
        """
        Load the complete model state, including BEATs and additional layers.
        """
        checkpoint = torch.load(load_path, map_location=device)
        
        num_labels = checkpoint['fc1_state_dict']['weight'].size(0)
        model = cls(num_labels=num_labels, 
                    l1_reg=checkpoint['l1_reg'], 
                    l2_reg=checkpoint['l2_reg'], 
                    freeze_encoder=checkpoint['freeze_encoder'])

        model.cfg = checkpoint['beats_config']
        model.beats = BEATs(model.cfg)
        model.beats.load_state_dict(checkpoint['beats_state_dict'])
        
        model.tcn.load_state_dict(checkpoint['tcn_state_dict'])
        model.fc1.load_state_dict(checkpoint['fc1_state_dict'])
        model.dropout1.load_state_dict(checkpoint['dropout1_state_dict'])

        model = model.to(device)
        
        print(f"Model loaded from {load_path}")
        return model, checkpoint