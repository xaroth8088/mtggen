from os.path import exists

import tensorflow as tf
from tqdm import tqdm
from tensorflow.data import TextLineDataset
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential, load_model

from sampling_module import generate_text
from vectorizer import load_vectorizer, build_vectorizer, save_vectorizer


def create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims):
    model = Sequential()

    # Add the Embedding layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dims,
        input_length=max_sequence_length,
        mask_zero=True
    ))

    # Add LSTM layers
    for _ in range(num_layers - 1):
        model.add(Bidirectional(LSTM(num_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(num_units)))

    # Add the Dense layer for output
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile the model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def on_epoch_end(epoch, model, vectorizer, sample_every_n_epochs):
    if sample_every_n_epochs <= 0:
        return

    if (sample_every_n_epochs > 1 and epoch == 0) or epoch % sample_every_n_epochs != 0:
        return

    print(f"\n----- Generating text after Epoch: {epoch + 1}")

    print(generate_text(100, model, vectorizer, 0.1, True))


def get_data_stats(data_path):
    max_sequence_length = 0
    num_lines = 0

    with open(data_path, 'r') as file:
        for line in file:
            caret_count = line.count('^')
            max_sequence_length = max(max_sequence_length, caret_count)
            num_lines += 1

    max_sequence_length += 1  # to account for the last token on each line

    print(f"Max seq len: {max_sequence_length}, Num lines: {num_lines}")

    return max_sequence_length, num_lines


def generate_sequences(sequence, max_sequence_length):
    sequence_length = tf.shape(sequence)[0]

    input_sequences = tf.TensorArray(dtype=tf.int16, size=0, dynamic_size=True)
    target_sequences = tf.TensorArray(dtype=tf.int16, size=0, dynamic_size=True)

    for i in tf.range(sequence_length - 1):
        input_seq = sequence[:i + 1]
        target_seq = sequence[i + 1]

        padding_size = max_sequence_length - tf.shape(input_seq)[0]
        input_seq_padded = tf.pad(input_seq, [[0, padding_size]], "CONSTANT")

        input_sequences = input_sequences.write(i, input_seq_padded)
        target_sequences = target_sequences.write(i, target_seq)

    return input_sequences.stack(), target_sequences.stack()


def preprocess_dataset(dataset, vectorize_layer, max_sequence_length, batch_size):
    # Vectorize the dataset
    dataset = dataset.map(vectorize_layer)

    # We definitely don't need 64-bit precision on this part, so go down to 16
    dataset = dataset.map(lambda x: tf.cast(x, tf.int16))

    # Generate sub-sequences
    dataset = dataset.flat_map(
        lambda sequence: tf.data.Dataset.from_tensor_slices(
            generate_sequences(sequence, max_sequence_length)
        )
    )

    num_lines = 0
    for _ in tqdm(dataset, desc="Counting dataset lines"):
        num_lines += 1

    # Setup for batching
    dataset = dataset.shuffle(num_lines).batch(batch_size)

    num_batches = 0
    for _ in dataset:
        num_batches += 1

    # Split it into training and validation sets
    train_size = int(0.8 * num_batches)
    training_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    return training_dataset, validation_dataset


def train_model(
        data_path=None,
        model_output_path=None,
        vectorizer_path=None,
        checkpoint_path=None,
        batch_size=None,
        num_units=None,
        num_layers=None,
        num_epochs=None,
        embedding_dims=None,
        sample_every_n_epochs=None
):
    max_sequence_length, num_lines = get_data_stats(data_path)

    dataset = TextLineDataset(data_path)

    if exists(checkpoint_path):
        print("Resuming training from saved checkpoint")
        model = load_model(checkpoint_path)
        vectorize_layer = load_vectorizer(vectorizer_path)
    else:
        vectorize_layer = build_vectorizer(dataset)
        save_vectorizer(vectorize_layer, vectorizer_path)

        print(vectorize_layer.get_vocabulary())
        print(f'Vocab len: {len(vectorize_layer.get_vocabulary())}')

        vocab_size = len(vectorize_layer.get_vocabulary())
        model = create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims)

    # Pre-process the dataset
    training_dataset, validation_dataset = preprocess_dataset(dataset, vectorize_layer, max_sequence_length, batch_size)

    # TODO: make patience configurable via CLI
    model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1
            ),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, model, vectorize_layer, sample_every_n_epochs)
            ),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    # Save the trained model
    model.save(
        model_output_path,
        save_format="keras"
    )
    print(generate_text(200, model, vectorize_layer, 0.1))
