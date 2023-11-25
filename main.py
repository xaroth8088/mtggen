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
    parser.add_argument('--sample_every_n_epochs', type=int, default=3,
                        help='Every n epochs, generate a short sample. (0 to disable)')
    parser.add_argument('--model_output_path', type=str, default='json_generator_model.keras',
                        help='Where to save the model')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (trade training speed for stability of training)')
    parser.add_argument('--num_units', type=int, default=128,
                        help='Width of the LSTM (trade "smarts" of the network for memory and training speed)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Depth of the LSTM (trade "smarts" of the network for memory and training speed)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='How many times to go through the training data (trade amount of learning for training time)')
    parser.add_argument('--embedding_dims', type=int, default=128,
                        help='Width of the embedding layer (trade understanding of complex relationships between words for memory and training speed)')
    args = parser.parse_args()

    if args.train:
        train_model(
            data_path=args.train,
            sample_every_n_epochs=args.sample_every_n_epochs,
            model_path=args.model_output_path,
            batch_size=args.batch_size,
            num_units=args.num_units,
            num_layers=args.num_layers,
            num_epochs=args.num_epochs,
            embedding_dims=args.embedding_dims
        )
    elif args.sample:
        sample_from_pretrained_model(
            model_path=args.model_path,
            characters_path="characters.txt"
        )
    else:
        print("Please specify either --train or --sample.")

if __name__ == "__main__":
    main()
