import argparse
import torchaudio
import torch
import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from src.utils.AudioDataset import AudioDataset4raw

def apply_augmentation(waveform, dataset, aug_types):
    augmentations = [
        ('time_stretch', dataset.time_stretching),
        ('pitch_shift', dataset.pitch_shifting),
        ('drc', dataset.dynamic_range_compression),
        ('noise', dataset.add_gaussian_noise),
        ('volume', dataset.adjust_volume)
    ]
    
    if aug_types == 'all':
        aug_name, aug_func = random.choice(augmentations)
    else:
        available_augs = [aug for aug in augmentations if aug[0] in aug_types]
        aug_name, aug_func = random.choice(available_augs)
    return aug_name, aug_func(waveform)

def process_audio_files(csv_path, output_dir, aug_per_file, aug_types, target_sample_rate=16000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    dataset = AudioDataset4raw(csv_path, target_sample_rate=target_sample_rate, augment=True)
    
    aug_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        audio_path = row['filename']
        label = row['label']
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        
        waveform = dataset.adjust_length(waveform)
        
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        original_output_path = os.path.join(output_dir, f"{base_filename}_original.wav")
        torchaudio.save(original_output_path, waveform, target_sample_rate)
        
        aug_data.append({
            'filename': original_output_path,
            'label': label
        })
        
        for i in range(aug_per_file):
            aug_name, aug_waveform = apply_augmentation(waveform, dataset, aug_types)
            
            aug_output_path = os.path.join(output_dir, f"{base_filename}_aug_{aug_name}_{i+1}.wav")
            torchaudio.save(aug_output_path, aug_waveform, target_sample_rate)
            
            aug_data.append({
                'filename': aug_output_path,
                'label': label
            })
    
    aug_df = pd.DataFrame(aug_data)
    aug_csv_path = os.path.join(output_dir, "augmented_audio_info.csv")
    aug_df.to_csv(aug_csv_path, index=False)
    print(f"Augmented audio information saved to {aug_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Augmentation")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save the augmented audio files")
    parser.add_argument("aug_per_file", type=int, help="Number of augmentations to create per original file")
    parser.add_argument("--aug_types", type=str, default='all', 
                        help="Augmentation types to apply. Use 'all' for all types, or specify types separated by commas. "
                             "Available types: time_stretch, pitch_shift, drc, noise, volume (default: all)")
    parser.add_argument("--seed", type=int, default=905, help="Random seed for reproducibility (default: 905)")
    args = parser.parse_args()

    aug_types = args.aug_types.split(',') if args.aug_types != 'all' else 'all'
    process_audio_files(args.csv_path, args.output_dir, args.aug_per_file, aug_types, seed=args.seed)
    print(f"Augmented audio files saved to {args.output_dir}")