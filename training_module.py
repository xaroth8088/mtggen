from os.path import exists

import numpy as np
from tensorflow.data import TextLineDataset
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence

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


class DataGenerator(Sequence):
    def __init__(self, dataset, num_lines, vectorizer, batch_size, max_sequence_length, validation_split=0.2,
                 is_training=True):
        self.dataset = dataset.cache().shuffle(num_lines) if is_training else dataset
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.max_length = max_sequence_length
        self.validation_split = validation_split
        self.split_index = int((1 - self.validation_split) * num_lines)
        self.total_lines = num_lines if is_training else int(num_lines * self.validation_split)
        self.dataset = self.dataset.batch(batch_size)

    def __len__(self):
        return int(np.ceil(self.total_lines / self.batch_size))

    def __getitem__(self, index):
        batch_data = next(iter(self.dataset.skip(index).take(1)))

        # Vectorize the text data on-the-fly
        texts = [text.numpy().decode('utf-8') for text in batch_data]
        sequences = self.vectorizer(texts)

        # Generate sequences
        input_sequences = [
            sequence[:i + 1]
            for sequence in sequences
            for i in range(1, len(sequence))
        ]

        input_sequences = pad_sequences(input_sequences, maxlen=self.max_length, padding='pre')
        x = input_sequences[:, :-1]
        y = input_sequences[:, -1]

        return x, y


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

    data_generator = DataGenerator(dataset, num_lines, vectorize_layer, batch_size, max_sequence_length,
                                   is_training=True)
    validation_data_generator = DataGenerator(dataset, num_lines, vectorize_layer, batch_size, max_sequence_length,
                                              validation_split=0.2, is_training=False)

    model.fit(
        data_generator,
        validation_data=validation_data_generator,
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
