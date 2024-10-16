import argparse
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from src.utils.AudioDataset import AudioDataset4raw

def visualize_waveform(waveform, title, ax):
    ax.plot(waveform.t().numpy())
    ax.set_title(title)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')

def visualize_spectrogram(waveform, sample_rate, title, ax):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128
    
    # 計算梅爾頻譜圖
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )(waveform)
    
    # 轉換為分貝
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    
    # 繪製頻譜圖
    img = ax.imshow(mel_spectrogram[0].numpy(), 
                    aspect='auto', 
                    origin='lower', 
                    extent=[0, waveform.shape[1]/sample_rate, 0, sample_rate/2],
                    cmap='viridis')
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    # 添加顏色條
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 設置 y 軸的刻度
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    ax.set_yticklabels(['0', '2k', '4k', '6k', '8k'])

def apply_and_visualize_augmentations(audio_path, output_dir):
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"raw sample_rate : {sample_rate}")
    target_sample_rate = 16000
    dataset = AudioDataset4raw('ASDor_wav/tmp.csv', target_sample_rate=target_sample_rate)  # We don't need a real CSV here
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    waveform = dataset.adjust_length(waveform)

    augmentations = [
        ('Original', lambda x: x),
        ('Time_Stretching', dataset.time_stretching),
        ('Pitch_Shifting', dataset.pitch_shifting),
        ('Dynamic_Range_Compression', dataset.dynamic_range_compression),
        ('Gaussian_Noise', dataset.add_gaussian_noise),
        ('Volume_Adjustment', dataset.adjust_volume)
    ]

    fig, axs = plt.subplots(len(augmentations), 2, figsize=(20, 7*len(augmentations)))
    fig.suptitle('Audio Augmentation Visualization', fontsize=16)

    # 創建保存增強音頻的目錄
    audio_output_dir = os.path.join(output_dir, 'augmented_audio')
    os.makedirs(audio_output_dir, exist_ok=True)

    for i, (name, aug_func) in enumerate(augmentations):
        augmented_waveform = aug_func(waveform)
        
        visualize_waveform(augmented_waveform, f'{name} - Waveform', axs[i, 0])
        visualize_spectrogram(augmented_waveform, target_sample_rate, f'{name} - Mel Spectrogram', axs[i, 1])

        # 保存增強後的音頻
        audio_output_path = os.path.join(audio_output_dir, f'{name}.wav')
        torchaudio.save(audio_output_path, augmented_waveform, target_sample_rate)
        print(f"Augmented audio saved to {audio_output_path}")

    plt.tight_layout()
    
    # 保存可視化圖像
    visualization_path = os.path.join(output_dir, 'augmentation_visualization.png')
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {visualization_path}")

    plt.close(fig)  # 關閉圖形以釋放內存

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize audio augmentations")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file")
    parser.add_argument("output_dir", type=str, help="Directory to save the output files")
    args = parser.parse_args()

    apply_and_visualize_augmentations(args.audio_path, args.output_dir)