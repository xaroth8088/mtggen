import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.saving import register_keras_serializable

TOKEN_DELIMITER = '^'
CARD_END = '$'
KEY_START = '@'
VALUE_START = '#'


@register_keras_serializable('mtggen')
def custom_splitter(text):
    return tf.strings.split(text, TOKEN_DELIMITER)


def build_vectorizer(dataset):
    vectorize_layer = TextVectorization(
        max_tokens=1500,
        standardize='lower',
        split=custom_splitter,
        output_mode='int',
        output_sequence_length=None
    )
    print("Building vectorizer... (this may take a while)")
    vectorize_layer.adapt(dataset)

    print(f'Vocab len: {len(vectorize_layer.get_vocabulary())}')

    return vectorize_layer


def save_vectorizer(vectorize_layer, vectorizer_path):
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string)
    processed_input = vectorize_layer(text_input)
    model = tf.keras.Model(text_input, processed_input)

    # Save the model
    model.save(vectorizer_path)


def load_vectorizer(vectorizer_path='vectorizer.keras'):
    loaded_model = tf.keras.models.load_model(vectorizer_path)
    return loaded_model.layers[1]
