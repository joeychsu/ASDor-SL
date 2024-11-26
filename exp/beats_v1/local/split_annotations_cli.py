import argparse
from src.utils.AudioDataset import split_annotations

def main():
    parser = argparse.ArgumentParser(description='Split annotations CSV file into training and validation sets.')
    parser.add_argument('csv_path', type=str, help='Path to the input annotations CSV file')
    parser.add_argument('train_csv_path', type=str, help='Path to save the training CSV file')
    parser.add_argument('valid_csv_path', type=str, help='Path to save the validation CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.82, 
                        help='Ratio of training data (default: 0.82)')
    parser.add_argument('--random_state', type=int, default=42, 
                        help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    split_annotations(
        args.csv_path,
        args.train_csv_path,
        args.valid_csv_path,
        train_ratio=args.train_ratio,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()