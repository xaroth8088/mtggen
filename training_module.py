import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import TextLineDataset
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TextVectorization
from tensorflow.keras.models import Sequential, load_model
from os.path import exists
from sampling_module import generate_text
from vectorizer import build_vectorizer
from tensorflow.keras.utils import Sequence


def create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims):
    layers = [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dims,
            input_length=max_sequence_length,
            mask_zero=True
        )
    ]

    for _ in range(num_layers - 1):
        layers.append(Bidirectional(LSTM(num_units, return_sequences=True)))

    layers.append(Bidirectional(LSTM(num_units)))

    layers.append(
        Dense(vocab_size, activation='softmax')
    )

    model = Sequential(layers)
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
    def __init__(self, dataset, num_lines, vectorizer, batch_size, max_sequence_length, validation_split=0.2, is_training=True):
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

    max_sequence_length += 1   # to account for the last token on each line

    print(f"Max seq len: {max_sequence_length}, Num lines: {num_lines}")

    return max_sequence_length, num_lines


def train_model(
        data_path='corpus/preprocessed_cards.txt',
        model_path='json_generator_model.keras',
        checkpoint_path='in_progress.keras',
        batch_size=16,
        num_units=128,
        num_layers=2,
        num_epochs=100,
        embedding_dims=128,
        sample_every_n_epochs=3
):
    max_sequence_length, num_lines = get_data_stats(data_path)

    dataset = TextLineDataset(data_path)

    vectorize_layer = build_vectorizer(dataset)

    if exists(checkpoint_path):
        print("Resuming training from saved checkpoint")
        model = load_model(checkpoint_path)
    else:
        vocab_size = len(vectorize_layer.get_vocabulary())
        model = create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims)

    model.summary()

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
        model_path,
        save_format="keras"
    )
    print(generate_text(100, model, vectorize_layer))
