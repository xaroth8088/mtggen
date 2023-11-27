import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from vectorizer import build_vectorizer


def sample_from_pretrained_model(
        model_path='json_generator_model.keras',
        data_path='mtg.jsonl',
        next_n_words=100,
        temperature=0.5
):
    # See note on vectorizer.py::custom_splitter() for more info on why we need this here
    with open(data_path, 'r', encoding='utf-8') as file:
        raw_text = np.array(file.readlines())

    vectorizer = build_vectorizer(raw_text)

    model = load_model(model_path)

    generated_text = generate_text(
        next_n_words,
        model,
        vectorizer,
        temperature=temperature,
        show_token_breaks=False
    )

    print(generated_text)


def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    sampled_index = np.argmax(np.random.multinomial(1, predictions, 1))
    return sampled_index


# TODO: we don't need 'next_n_words', and instead should be a safety limit, since we should just generate until we
#       get to the end of a JSON object.
def generate_text(next_n_words, model, vectorizer, temperature=0.1, show_token_breaks=False):
    # TODO: allow a key/value pair to be a seed (this will be more complex due to splitting/tokenization process)
    context_sequence = vectorizer(['{'])
    max_sequence_length = model.layers[0].input_length
    context_sequence = pad_sequences(context_sequence, maxlen=max_sequence_length, padding='pre')
    output_sequence = list(context_sequence[0])  # TODO: output_sequence doesn't need all the leading 0's :P

    for _ in range(next_n_words):
        # Call model.predict() to get the prediction weights
        predictions = model.predict(context_sequence, verbose=0)[0]

        # Use the prediction to select the next character index
        predicted_index = sample_with_temperature(predictions, temperature)

        # Append that index to the end of output_sequence
        output_sequence.append(predicted_index)

        # Advance the context by one
        context_sequence = np.append(context_sequence, predicted_index)[-(max_sequence_length):].reshape(1, -1)

    vocabulary = vectorizer.get_vocabulary()
    join_string = ''
    if show_token_breaks:
        join_string = ' | '
    generated_text = join_string.join([
        vocabulary[index]
        for index in output_sequence
    ])

    return generated_text
