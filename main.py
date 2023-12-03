import argparse

from sampling_module import sample_from_pretrained_model
from training_module import train_model
from webserver_module import start_server

def main():
    parser = argparse.ArgumentParser(description='Train or sample from a JSON generator model.')

    training_args = parser.add_argument_group('Training Options')
    sampling_args = parser.add_argument_group('Sampling Options')
    web_args = parser.add_argument_group('Webserver Options')
    common_args = parser.add_argument_group('Common Options')

    # Training params
    training_args.add_argument('--train', action='store', metavar='data_path', type=str,
                        help='Train the model using the specified data file')
    training_args.add_argument('--checkpoint_path', type=str, default='in_progress.keras',
                        help='After each epoch, a checkpoint will be saved here.  If that file already exists, training will resume from that point')
    training_args.add_argument('--sample_every_n_epochs', type=int, default=3,
                        help='Every n epochs, generate a short sample. (0 to disable)')
    training_args.add_argument('--model_output_path', type=str, default='json_generator_model.keras',
                        help='Where to save the model')
    training_args.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (trade training speed for stability of training)')
    training_args.add_argument('--num_units', type=int, default=128,
                        help='Width of the LSTM (trade "smarts" of the network for memory and training speed)')
    training_args.add_argument('--num_layers', type=int, default=1,
                        help='Depth of the LSTM (trade "smarts" of the network for memory and training speed)')
    training_args.add_argument('--num_epochs', type=int, default=100,
                        help='How many times to go through the training data (trade amount of learning for training time)')
    training_args.add_argument('--embedding_dims', type=int, default=128,
                        help='Width of the embedding layer (trade understanding of complex relationships between words for memory and training speed)')

    # Sampling and web params
    sampling_args.add_argument('--sample', action='store', metavar='data_path', type=str,
                        help='Sample from a previously-trained model.  The path to the data that was used for training is also required for now.')
    sampling_args.add_argument('--temperature', type=float, default=0.5,
                        help='How creative will the generation be (range: 0.0 to 1.0; lower numbers are less creative)')

    # Web server params
    web_args.add_argument('--web', action='store', metavar='data_path', type=str,
                               help='Start a web server on port 8000 with a single route at / that returns JSON suitable for use in the card HTML')

    # Used in all
    common_args.add_argument('--model_path', type=str, default='json_generator_model.keras',
                        help='Path to the model file for sampling')

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
            embedding_dims=args.embedding_dims,
            checkpoint_path=args.checkpoint_path
        )
    elif args.sample:
        sample_from_pretrained_model(
            model_path=args.model_path,
            data_path=args.sample,
            temperature=args.temperature,
        )
    elif args.web:
        start_server(
            model_path=args.model_path,
            data_path=args.web,
            temperature=args.temperature
        )
    else:
        print("Please specify either --train or --sample.")


if __name__ == "__main__":
    main()
