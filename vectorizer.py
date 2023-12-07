import tensorflow as tf
from tensorflow.keras.layers import TextVectorization


def custom_splitter(text):
    return tf.strings.split(text, '^')


def build_vectorizer(dataset):
    # TODO: save/load the vectorizer, so that we're not reconstructing it every single time
    vectorize_layer = TextVectorization(
        max_tokens=1500,
        standardize='lower',
        split=custom_splitter,
        output_mode='int',
        output_sequence_length=None
    )
    vectorize_layer.adapt(dataset)

    print(vectorize_layer.get_vocabulary())
    print(f'Vocab len: {len(vectorize_layer.get_vocabulary())}')

    return vectorize_layer
