from json import loads
from re import findall, sub

import numpy as np
import tensorflow as tf
from tensorflow import py_function
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, TextVectorization, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences, Sequence

from rules_templates import rules_templates
from sampling_module import generate_text


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


def custom_splitter(text):
    retval = []
    for raw_line in text.numpy():
        line = raw_line.decode('utf-8')

        # Start the object
        tokens = ['{']

        # Handle the case where we're seeding a new object
        if line == '{':
            retval.append(tokens)
            continue

        # TODO: explode out the \w+|\W thing so that we can have a little more control on what counts as a word barrier
        #       (e.g. apostrophes are considered a word barrier right now, which isn't desirable)

        json = loads(line)
        for key, value in json.items():
            tokens.append(f'"{key}":')
            if isinstance(value, list):
                if len(value) == 0:
                    tokens.append('[]')
                    continue

                tokens.append('["')
                list_tokens = []
                for element in value:
                    list_tokens.extend(findall(r'\w+|\W', str(element)))
                    list_tokens.append('","')
                list_tokens[-1] = '"],'
                tokens.extend(list_tokens)
            else:
                # Strip reminder text (TODO: we don't need this once the data prep script does it)
                tidied_value = sub(r'\(.+?\)', '', str(value))

                tokens.append('"')
                # Let's let names be arbitrary constructions, instead of copied words
                if key == 'name':
                    tokens.extend(list(value))
                else:
                    # TODO: wrap the rules templates in _checks_ for \W on either side, but
                    #       don't _capture_ the \W's
                    regex = (
                            r'|'.join(rules_templates)  # rules templates
                            + r'|\{.+?}'  # mana symbols
                            + r'|\w+'  # whole words
                            + r'|\W'  # non-word characters
                    )
                    tokens.extend(findall(regex, tidied_value))
                tokens.append('",')

        # End the object
        last_token = tokens[-1]
        last_token = last_token[:-1]  # Remove the comma
        last_token += "}\n"  # Close the object
        tokens[-1] = last_token
        retval.append(list(tokens))

    # Ensure that all inner lists have the same length
    max_len = max(len(tokens) for tokens in retval)
    retval = [tokens + [''] * (max_len - len(tokens)) for tokens in retval]

    return tf.constant(retval, dtype=tf.string)


def train_model(
        data_path='mtg.jsonl',
        model_path='json_generator_model.keras',
        batch_size=1,
        num_units=128,
        num_layers=1,
        num_epochs=100,
        embedding_dims=128,
        sample_every_n_epochs=3
):
    with open(data_path, 'r', encoding='utf-8') as file:
        raw_text = np.array(file.readlines())

    vectorizer = TextVectorization(
        max_tokens=1500,
        standardize='lower',
        split=lambda x: py_function(custom_splitter, [x], Tout=tf.string),
        output_mode='int',
    )
    vectorizer.adapt(raw_text)

    print(vectorizer.get_vocabulary())
    print(f'Vocab len: {len(vectorizer.get_vocabulary())}')

    sequences = vectorizer(raw_text)

    # Find the maximum sequence length
    max_sequence_length = max(len(seq) for seq in sequences)
    print(f"Max seq len: {max_sequence_length}")

    vocab_size = len(vectorizer.get_vocabulary())

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
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, model, vectorizer, sample_every_n_epochs)
            ),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )

    # Save the trained model
    model.save(model_path)
    print(generate_text(100, model, vectorizer))
