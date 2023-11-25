import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_generated_json(raw_string):
    just_json = raw_string.split("@")[0]
    return just_json


def sample_from_model(temperature=0.5, max_length=400, chars=None, model=None):
    temperature = 0.0
    max_length = 15
    prefix_text = "Hell"

    char_indices = {char: i for i, char in enumerate(chars)}
    indices_char = {value: key for key, value in char_indices.items()}

    # Initialize the input tensor with the seed sequence
    x_pred = np.zeros((1, len(prefix_text), len(char_indices)), dtype=bool)
    for t, char in enumerate(prefix_text):
        x_pred[0, t, char_indices[char]] = 1.0

    generated_text = prefix_text
    for _ in range(max_length):
        # Predict the next character probabilities
        preds = model.predict(x_pred, verbose=0)[0]

        print("Seed Sequence:", prefix_text)
        print("Predicted Probabilities:", preds)
        print("Generated Sequence:", generated_text)

        # Sample the next character index based on the temperature
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        # Append the next character to the generated text
        generated_text += next_char

        # Update the input sequence for the next iteration
        x_pred = np.zeros((1, len(generated_text), len(char_indices)), dtype=bool)
        for t, char in enumerate(generated_text[-len(prefix_text):]):  # Use the last part of generated_text
            x_pred[0, t, char_indices[char]] = 1.0
        # for t, char in enumerate(generated_text):
        #     x_pred[0, t, char_indices[char]] = 1.0

    return clean_generated_json(generated_text)


def sample_from_pretrained_model(model_path, characters_path, max_length=400, temperature=0.5):
    model = load_model(model_path)

    with open(characters_path, 'r') as file:
        characters = file.read()

    generated_text = sample_from_model(
        temperature=temperature,
        max_length=max_length,
        chars=characters,
        model=model
    )

    print(generated_text)


def sample(preds, temperature=1.0):
    temperature = 0.0
    preds = np.asarray(preds).astype('float64')
    print("********************************************************")
    print(preds)

    # Ensure temperature is not exactly 0.0 to avoid division by zero
    if temperature > 0.0:
        preds = np.log(np.maximum(preds, 1e-10)) / temperature
    else:
        preds = np.log(np.maximum(preds, 1e-10))
    print(preds)

    exp_preds = np.exp(preds)
    print(preds)
    preds = exp_preds / np.sum(exp_preds)
    print(preds)

    probas = np.random.multinomial(1, preds, 1)
    print(np.argmax(probas))
    print("********************************************************")
    return np.argmax(probas)

#---

def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    sampled_index = np.argmax(np.random.multinomial(1, predictions, 1))
    return sampled_index


def generate_text(seed_text, next_n_words, model, vectorizer, temperature=0.1):
    context_sequence = vectorizer([seed_text])
    max_sequence_length = model.layers[0].input_length
    context_sequence = pad_sequences(context_sequence, maxlen=max_sequence_length, padding='pre')
    output_sequence = list(context_sequence[0])

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
    generated_text = ''.join([
        vocabulary[index]
        for index in output_sequence
    ])

    return generated_text
