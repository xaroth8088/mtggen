import argparse

from sampling_module import sample_from_pretrained_model
from training_module import train_model
from webserver_module import start_server


def add_sampling_common_args(subparser):
    subparser.add_argument("model_path", type=str,
                           help="Path to the model file for sampling")
    subparser.add_argument('--max_output_tokens', type=int, default=200,
                           help='How many tokens should we try to generate before giving up on making a complete JSON object?')
    subparser.add_argument('--temperature', type=float, default=0.5,
                           help='How creative will the generation be (range: 0.0 to 1.0; lower numbers are less creative)')


def main():
    parser = argparse.ArgumentParser(description='Train or sample from a JSON generator model.')

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Training params
    parser_train = subparsers.add_parser("train", help="Training mode")

    parser_train.add_argument("--data_path", type=str, default='corpus/preprocessed_cards.txt',
                              help="Path to the data file for training")
    parser_train.add_argument('--model_output_path', type=str, default='mtggen.keras',
                               help='Where to save the fully-trained model')
    parser_train.add_argument('--checkpoint_path', type=str, default='in_progress.keras',
                               help='After each epoch, a checkpoint will be saved here.  If that file already exists, training will resume from that point')
    parser_train.add_argument('--num_epochs', type=int, default=100,
                               help='How many times to go through the training data (trade amount of learning for training time)')
    parser_train.add_argument('--sample_every_n_epochs', type=int, default=0,
                               help='Every n epochs, generate a short sample. (0 to disable)')
    parser_train.add_argument('--batch_size', type=int, default=16,
                               help='Batch size for training (trade training speed for stability of training)')
    parser_train.add_argument('--num_units', type=int, default=150,
                               help='Width of the LSTM (trade "smarts" of the network for memory and training speed)')
    parser_train.add_argument('--num_layers', type=int, default=2,
                               help='Depth of the LSTM (trade "smarts" of the network for memory and training speed)')
    parser_train.add_argument('--embedding_dims', type=int, default=128,
                               help='Width of the embedding layer (trade understanding of complex relationships between words for memory and training speed)')

    # Sampling and web params
    parser_sample = subparsers.add_parser("sample", help="Sample from a previously-trained model")
    add_sampling_common_args(parser_sample)

    # Web server params
    parser_web = subparsers.add_parser("web", help='Start a web server. /html/card.html to see a rendered card')
    add_sampling_common_args(parser_web)
    parser_web.add_argument('--listen_address', type=str, default='localhost',
                            help='Host or IP to listen on.  Use 0.0.0.0 to serve publicly, though this is strongly contraindicated')
    parser_web.add_argument('--port', type=int, default=8000,
                            help='Port to listen on')


    # Used in all
    for subparser in [parser_train, parser_sample, parser_web]:
        subparser.add_argument('--vectorizer_path', type=str, default='vectorizer.keras',
                             help='Path to the serialized vectorizer file')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(
            data_path=args.data_path,
            model_output_path=args.model_output_path,
            vectorizer_path=args.vectorizer_path,
            sample_every_n_epochs=args.sample_every_n_epochs,
            batch_size=args.batch_size,
            num_units=args.num_units,
            num_layers=args.num_layers,
            num_epochs=args.num_epochs,
            embedding_dims=args.embedding_dims,
            checkpoint_path=args.checkpoint_path,
        )
    elif args.mode == 'sample':
        sample_from_pretrained_model(
            model_path=args.model_path,
            temperature=args.temperature,
            vectorizer_path=args.vectorizer_path,
            max_output_tokens=args.max_output_tokens,
        )
    elif args.mode == 'web':
        start_server(
            model_path=args.model_path,
            temperature=args.temperature,
            vectorizer_path=args.vectorizer_path,
            max_output_tokens=args.max_output_tokens,
            listen_address=args.listen_address,
            port=args.port
        )


if __name__ == "__main__":
    main()
