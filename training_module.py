import numpy as np
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import pad_sequences, Sequence
from os.path import exists
from sampling_module import generate_text
from vectorizer import build_vectorizer


def create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims):
    layers = [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dims,
            input_length=max_sequence_length - 1,
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
    def __init__(self, data, vectorizer, batch_size, max_length, validation_split=0.2):
        self.data = data
        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.max_length = max_length
        self.indexes = np.arange(len(self.data))
        np.random.shuffle(self.indexes)
        self.split_index = int((1 - self.validation_split) * len(self.data))

    def __len__(self):
        if self.validation_split == 0.0:
            return int(np.ceil(len(self.data) / self.batch_size))
        else:
            return int(np.ceil(self.split_index / self.batch_size))

    def __getitem__(self, index):
        if self.validation_split == 0.0:
            batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        else:
            batch_indexes = self.indexes[
                            index * self.batch_size:min((index + 1) * self.batch_size, self.split_index)
                            ]
        batch_data = self.data[batch_indexes]
        sequences = self.vectorizer(batch_data)
        input_sequences = [
            sequence[:i + 1]
            for sequence in sequences
            for i in range(1, len(sequence))
        ]
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_length, padding='pre')
        x = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        return x, y


def train_model(
        data_path='mtg.jsonl',
        model_path='json_generator_model.keras',
        checkpoint_path='in_progress.keras',
        batch_size=1,
        num_units=128,
        num_layers=1,
        num_epochs=100,
        embedding_dims=128,
        sample_every_n_epochs=3
):
    with open(data_path, 'r', encoding='utf-8') as file:
        raw_text = np.array(file.readlines())

    vectorizer = build_vectorizer(raw_text)

    sequences = vectorizer(raw_text)

    # Find the maximum sequence length
    max_sequence_length = max(len(seq) for seq in sequences)
    print(f"Max seq len: {max_sequence_length}")

    vocab_size = len(vectorizer.get_vocabulary())

    if exists(checkpoint_path):
        print("Resuming training from saved checkpoint")
        model = load_model(checkpoint_path)
    else:
        model = create_model(vocab_size, max_sequence_length, num_units, num_layers, embedding_dims)

    model.summary()

    data_generator = DataGenerator(raw_text, vectorizer, batch_size, max_length=max_sequence_length)
    validation_data_generator = DataGenerator(raw_text, vectorizer, batch_size, max_length=max_sequence_length,
                                              validation_split=0.2)

    model.fit(
        data_generator,
        epochs=num_epochs,
        validation_data=validation_data_generator,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1
            ),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, model, vectorizer, sample_every_n_epochs)
            ),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    # Save the trained model
    model.save(
        model_path,
        save_format="keras"
    )
    print(generate_text(100, model, vectorizer))
