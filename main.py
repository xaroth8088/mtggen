import argparse
from training_module import train_model
from sampling_module import sample_from_pretrained_model

def main():
    parser = argparse.ArgumentParser(description='Train or sample from a JSON generator model.')
    parser.add_argument('--train', action='store', metavar='data_path', type=str,
                        help='Train the model using the specified data file')
    parser.add_argument('--sample', action='store_true', help='Sample from a previously-trained model')
    parser.add_argument('--model_path', type=str, default='json_generator_model.keras',
                        help='Path to the model file for sampling')
    args = parser.parse_args()

    if args.train:
        train_model(data_path=args.train)
    elif args.sample:
        sample_from_pretrained_model(
            model_path=args.model_path,
            characters_path="characters.txt"
        )
    else:
        print("Please specify either --train or --sample.")

if __name__ == "__main__":
    main()
