import json
import re

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from titlecase import titlecase

from preprocess_cards import replace_tilde_with_card_name, json_walker
from vectorizer import load_vectorizer, CARD_END, VALUE_START, KEY_START


def sample_from_pretrained_model(
        max_output_tokens=None,
        model_path=None,
        vectorizer_path=None,
        temperature=None
):
    vectorize_layer = load_vectorizer(vectorizer_path)

    model = load_model(model_path)

    for _ in range(1):
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
    # TODO: allow a key/value pair to be a seed (if the network is trained with the key/values appearing in random order, this will be simple to do!)
    context_sequence = vectorizer([KEY_START])
    end_token = vectorizer([CARD_END])[0].numpy()[0]
    output_sequence = list(context_sequence[0].numpy())
    max_sequence_length = model.layers[0].input_length
    context_sequence = pad_sequences(context_sequence, maxlen=max_sequence_length, padding='pre')

    while len(output_sequence) < max_output_tokens and output_sequence[-1:][0] != end_token:
        # Call model.predict() to get the prediction weights
        predictions = model.predict(context_sequence)[0]

        # Use the prediction to select the next character index
        predicted_index = sample_with_temperature(predictions, temperature)

        # Append that index to the end of output_sequence
        output_sequence.append(predicted_index)

        # Advance the context by one
        context_sequence = np.append(context_sequence, predicted_index)[-(max_sequence_length):].reshape(1, -1)

    print("max_output_tokens:", max_output_tokens, "len:", len(output_sequence), "end of output_sequence:",
          output_sequence[-1:][0], "vs:", end_token)

    if output_sequence[-1:][0] == end_token:
        output_sequence.pop()

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

    if show_token_breaks is False:
        card = convert_to_dict(generated_text)
        card = pretty_format_card(card)
        return json.dumps(card)

    return generated_text


def convert_to_dict(tokens):
    """
    Converts a list of tokens into a dictionary. Tokens are grouped into key-value pairs
    based on specific start tokens for keys and values.

    :param tokens: List of tokens to convert.
    :return: Dictionary with grouped tokens.
    """
    result = {}
    current_string = ""
    key = None

    for item in tokens:
        if item in {KEY_START, VALUE_START}:
            current_string = current_string.strip()
            if key is not None:
                result.setdefault(key, []).append(current_string)
                key = None if item == KEY_START else key
            elif current_string:
                key = current_string
            current_string = ""
        else:
            current_string += item

    if key and current_string.strip():
        result.setdefault(key, []).append(current_string.strip())

    return result


def pretty_format_card(card):
    # some keys _only_ have flat text, so convert those here
    for key in ["name", "text", "power", "toughness", "manacost", "rarity"]:
        if key in card:
            card[key] = '\n'.join(card[key])

    card = json_walker(card, title_case, True)

    if "text" in card:
        card["text"] = card["text"].lower()
        card["text"] = sentence_case(card["text"])
        card["text"] = replace_tilde_with_card_name(card["text"], card["name"])

    return card


def title_case(text):
    return titlecase(text)


def sentence_case(text):
    sentence_pattern = r'([^.!:?]*)([.!:?]|\Z)'

    def capitalize_sentence(match):
        sentence, punct = match.groups()
        return re.sub(r'(\S)', lambda x: x.group(1).upper(), sentence, 1) + (punct if punct else '')

    lines = text.split('\n')
    capitalized_lines = [''.join(capitalize_sentence(match) for match in re.finditer(sentence_pattern, line, re.DOTALL)) for line in lines]

    return '\n'.join(capitalized_lines)


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
