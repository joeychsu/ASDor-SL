import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import os
import random
import torch.nn.functional as F
import sys
from sklearn.model_selection import train_test_split

# 創建 annotations.csv
def create_annotations_csv_by_folder(wav_dir, output_csv):
    file_list = []
    labels = []
    other_files = []

    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith(".wav"):
                relative_path = os.path.relpath(os.path.join(root, file), wav_dir)
                full_path = os.path.join(wav_dir, relative_path)
                
                if "enroll_wav" in relative_path:
                    file_list.append(full_path)
                    labels.append("ok")
                elif "verify_ng_wav" in relative_path:
                    file_list.append(full_path)
                    labels.append("ng")
                elif "verify_ok_wav" in relative_path:
                    file_list.append(full_path)
                    labels.append("ok")
                elif "bg_wav" in relative_path:
                    file_list.append(full_path)
                    labels.append("other")
                elif "other_wav" in relative_path:
                    other_files.append(full_path)
                else:
                    # 對於不在特定資料夾中的檔案，使用原來的邏輯
                    if "ok" in file.lower():
                        file_list.append(full_path)
                        labels.append("ok")
                    elif "ng" in file.lower():
                        file_list.append(full_path)
                        labels.append("ng")
                    else:
                        file_list.append(full_path)
                        labels.append("other")

    df = pd.DataFrame({"filename": file_list, "label": labels})
    df.to_csv(output_csv, index=False)
    print(f"Annotations saved to {output_csv}")
    
    print("\nFiles in other_wav folder (not included in csv):")
    for file in other_files:
        print(file)

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
    def __init__(self, annotations_file, target_sample_rate=16000, target_length=5, 
                 augment=True, augment_type='None', augment_prob=0.5, device='cuda'):
        self.audio_labels = pd.read_csv(annotations_file)
        self.target_sample_rate = target_sample_rate
        self.target_length = target_sample_rate * target_length
        self.augment = augment
        self.augment_type = augment_type
        self.augment_prob = augment_prob
        self.label_mapping = {"ok": 0, "ng": 1, "other": 2}
        self.device = device

        # 初始化转换器
        self.spectrogram = T.Spectrogram(n_fft=400, power=None)
        self.time_stretch = T.TimeStretch(n_freq=201)
        self.griffin_lim = T.GriffinLim(n_fft=400)

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = self.audio_labels.iloc[idx, 0]
        label = self.label_mapping[self.audio_labels.iloc[idx, 1]]

        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 確保音頻長度一致（在增強之前）
        waveform = self.adjust_length(waveform)

        if self.augment:
            waveform = self.apply_augmentation(waveform)

        # 最终检查和调整
        waveform = self.adjust_length(waveform)

        return waveform.to(self.device).squeeze(0), label

    def adjust_length(self, waveform):
        if waveform.shape[1] > self.target_length:
            # 隨機裁剪
            start = random.randint(0, waveform.shape[1] - self.target_length)
            waveform = waveform[:, start:start + self.target_length]
        elif waveform.shape[1] < self.target_length:
            # 填充
            waveform = F.pad(waveform, (0, self.target_length - waveform.shape[1]))
        return waveform

    def apply_augmentation(self, waveform):
        if not self.augment:
            return waveform

        if random.random() < self.augment_prob:
            if self.augment_type.lower() == 'none':
                # 如果没有指定增强类型，随机选择一种
                augmentation_choice = random.choice([
                    self.time_stretching,
                    self.pitch_shifting,
                    self.dynamic_range_compression,
                    self.add_gaussian_noise,
                    self.adjust_volume
                ])
            else:
                # 使用指定的增强类型
                augmentation_choice = getattr(self, self.augment_type, None)
                if augmentation_choice is None:
                    print(f"Warning: Specified augmentation type '{self.augment_type}' not found. No augmentation applied.")
                    return waveform

            return augmentation_choice(waveform)
        else : 
            return waveform

    def time_stretching(self, waveform, stretch_factor=None):
        if stretch_factor is None:
            stretch_factor = random.uniform(0.8, 1.2)
        
        spec = self.spectrogram(waveform)
        stretched_spec = self.time_stretch(spec, stretch_factor)
        return self.griffin_lim(stretched_spec.abs())

    def pitch_shifting(self, waveform, n_steps=None):
        if n_steps is None:
            n_steps = random.uniform(-4, 4)
        return torchaudio.functional.pitch_shift(waveform, self.target_sample_rate, n_steps)

    def dynamic_range_compression(self, waveform, threshold=-10, ratio=4):
        db = torchaudio.functional.amplitude_to_DB(
            waveform, 
            multiplier=20.0, 
            amin=1e-10, 
            db_multiplier=1.0, 
            top_db=80.0
        )
        compressed_db = torch.where(db > threshold, threshold + (db - threshold) / ratio, db)
        return torchaudio.functional.DB_to_amplitude(
            compressed_db, 
            ref=1.0, 
            power=0.5
        )

    def add_gaussian_noise(self, waveform, snr_db=10):
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def adjust_volume(self, waveform, min_gain_db=-10, max_gain_db=10):
        gain = random.uniform(min_gain_db, max_gain_db)
        return torchaudio.functional.gain(waveform, gain_db=gain)

