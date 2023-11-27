from json import loads
from re import findall, sub

import tensorflow as tf
from tensorflow import py_function
from tensorflow.keras.layers import TextVectorization
from rules_templates import rules_templates


# NOTE: while py_function() is generally discouraged, I can't find a way to examine and manipulate arbitrary JSON
#       from within a graph function, because you're not allowed to "see" the raw string inside a SymbolicTensor.
#       Until and unless this gets puzzled out, we'll need to keep this in eager execution mode, we won't be able
#       to properly serialize the TextVectorizer layer, and overall training time will be slowed.
@py_function(Tout=tf.string)
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


def build_vectorizer(raw_text):
    vectorizer = TextVectorization(
        max_tokens=1500,
        standardize='lower',
        split=custom_splitter,
        output_mode='int',
    )
    vectorizer.adapt(raw_text)

    print(vectorizer.get_vocabulary())
    print(f'Vocab len: {len(vectorizer.get_vocabulary())}')
    return vectorizer
