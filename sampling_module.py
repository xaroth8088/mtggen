import json
import re

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess_cards import replace_tilde_with_card_name, json_walker
from vectorizer import load_vectorizer


def sample_from_pretrained_model(
        max_output_tokens=None,
        model_path=None,
        vectorizer_path=None,
        temperature=None
):
    vectorize_layer = load_vectorizer(vectorizer_path)

    model = load_model(model_path)

    for _ in range(5):
        generated_text = generate_text(
            max_output_tokens,
            model,
            vectorize_layer,
            temperature=temperature,
            show_token_breaks=False
        )
        print("***********************************************************")
        print(generated_text)


def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    with np.errstate(divide='ignore'):
        predictions = np.log(predictions)
    predictions = predictions / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    sampled_index = np.argmax(np.random.multinomial(1, predictions, 1))
    return sampled_index


def generate_text(max_output_tokens, model, vectorizer, temperature, show_token_breaks=False):
    # TODO: allow a key/value pair to be a seed (this will be more complex due to splitting/tokenization process)
    context_sequence = vectorizer(['{'])
    end_token = vectorizer(['"]}'])[0].numpy()[0]
    output_sequence = list(context_sequence[0].numpy())
    max_sequence_length = model.layers[0].input_length
    context_sequence = pad_sequences(context_sequence, maxlen=max_sequence_length, padding='pre')

    while len(output_sequence) < max_output_tokens and output_sequence[-1:][0] != end_token:
        # Call model.predict() to get the prediction weights
        predictions = model.predict(context_sequence, verbose=0)[0]

        # Use the prediction to select the next character index
        predicted_index = sample_with_temperature(predictions, temperature)

        # Append that index to the end of output_sequence
        output_sequence.append(predicted_index)

        # Advance the context by one
        context_sequence = np.append(context_sequence, predicted_index)[-(max_sequence_length):].reshape(1, -1)

    print("max_output_tokens:", max_output_tokens, "len:", len(output_sequence), "end of output_sequence:", output_sequence[-1:][0], "vs:", end_token)

    return unvectorize(output_sequence, vectorizer, show_token_breaks)


def unvectorize(output_sequence, vectorizer, show_token_breaks):
    vocabulary = vectorizer.get_vocabulary()

    join_string = ''

    if show_token_breaks:
        join_string = ' | '

    generated_text = join_string.join([
        vocabulary[index]
        for index in output_sequence
    ])

    generated_text = generated_text.replace('\n', '\\n')

    if show_token_breaks is False:
        try:
            card = extract_valid_json(generated_text)
        except (ValueError, json.JSONDecodeError):
            print("Unable to generate card.  Generated text was:", generated_text)
            card = {}

        card = pretty_format_card(card)
        return json.dumps(card)

    return generated_text


def pretty_format_card(card):
    card = json_walker(card, title_case, True)

    if "text" in card:
        card["text"] = sentence_case(card["text"])
        card["text"] = replace_tilde_with_card_name(card["text"], card["name"])

    return card


def title_case(text):
    return text.title()


def sentence_case(text):
    return re.sub(r'\w[\w ]+', lambda x: x[0].capitalize(), text)


def extract_valid_json(input_string):
    # Define a custom JSONObject to find the end of the JSON object
    class JSONObject(json.JSONDecoder):
        def decode(self, s, _w=json.decoder.WHITESPACE.match):
            obj, end = super().raw_decode(s, idx=_w)
            return obj, end

    # Create an instance of the custom decoder
    decoder = JSONObject()

    # Decode the input string
    valid_json, end_index = decoder.raw_decode(input_string)

    return valid_json
