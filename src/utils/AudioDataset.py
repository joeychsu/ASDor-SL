import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as transforms
import os
import random
import torch.nn.functional as F
import sys
from sklearn.model_selection import train_test_split

# 創建 annotations.csv
def create_annotations_csv_by_folder(wav_dir, output_csv):
    file_list = []
    labels = []
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith(".wav"):
                relative_path = os.path.relpath(os.path.join(root, file), wav_dir)
                file_list.append(os.path.join(wav_dir, relative_path))
                if "ok" in file.lower():
                    label = "ok"
                elif "ng" in file.lower():
                    label = "ng"
                else:
                    label = "other"
                labels.append(label)
    df = pd.DataFrame({"filename": file_list, "label": labels})
    df.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")

def split_annotations(csv_path, train_csv_path, valid_csv_path, train_ratio=0.82, random_state=42):
    assert train_ratio < 1.0

    # 讀取 CSV 文件
    df = pd.read_csv(csv_path)

    # 首先分割出訓練集
    train_df, valid_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state, stratify=df['label'])
        
    # 保存分割後的 CSV 文件
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)

    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Validation samples: {len(valid_df)} ({len(valid_df)/len(df):.2%})")

    # 打印每個集合中各類別的數量
    for name, dataset in [('Training', train_df), ('Validation', valid_df)]:
        print(f"\n{name} set class distribution:")
        print(dataset['label'].value_counts(normalize=True))

class AudioDataset4raw(Dataset):
    def __init__(self, annotations_file, target_sample_rate=16000, target_length=10, augment=True):
        self.audio_labels = pd.read_csv(annotations_file)
        self.target_sample_rate = target_sample_rate
        self.target_length = target_sample_rate * target_length  # 10 秒 * 16000 採樣率 = 160000
        self.augment = augment
        self.label_mapping = {"ok": 0, "ng": 1, "other": 2}

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = self.audio_labels.iloc[idx, 0]
        label = self.label_mapping[self.audio_labels.iloc[idx, 1]]

        # 加載音頻
        waveform, sample_rate = torchaudio.load(audio_path)

        # 重採樣（如果需要）
        if sample_rate != self.target_sample_rate:
            resample_transform = transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resample_transform(waveform)

        # 確保單聲道音頻
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        # 應用音頻增強（如果啟用）
        if self.augment:
            waveform = self.apply_augmentation(waveform)

        # 確保音頻長度一致
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            waveform = F.pad(waveform, (0, self.target_length - waveform.shape[1]))

        return waveform.squeeze(0), label

    def apply_augmentation(self, waveform):
        # 時間偏移
        if random.random() < 0.1:
            waveform = self.time_shift(waveform)

        # 添加合成噪音
        if random.random() < 0.1:
            waveform = self.add_noise(waveform)

        # 音量調整
        if random.random() < 0.1:
            waveform = waveform * random.uniform(0.8, 1.2)

        return waveform

    def time_shift(self, waveform, shift_limit=0.1):
        shift = int(random.uniform(-shift_limit, shift_limit) * waveform.shape[1])
        return torch.roll(waveform, shifts=shift, dims=1)

    def add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform)
        return waveform + noise_level * noise
