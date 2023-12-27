import argparse
import re
from json import load
from re import findall, sub, escape

from tqdm import tqdm
from unidecode import unidecode

from rules_templates import rules_templates


def filter_unwanted_keys(card):
    for key in [
        "sourceProducts", "setCode", "purchaseUrls", "foreignData", "artist", "artistIds",
        "availability", "boosterTypes", "colorIdentity", "colors", "convertedManaCost",
        "edhrecRank", "edhrecSaltiness", "finishes", "frameVersion", "hasFoil", "hasNonFoil",
        "identifiers", "isReprint", "layout", "legalities", "manaValue", "number",
        "originalText", "originalType", "printings", "type", "uuid", "variations",
        "flavorText", "borderColor", "language", "rulings", "asciiName", "cardParts",
        "colorIndicator", "duelDeck", "faceConvertedManaCost", "faceFlavorName",
        "faceManaValue", "faceName", "flavorName", "frameEffects", "hasAlternativeDeckLimit",
        "isFullArt", "isOnlineOnly", "isPromo", "isStarter", "isStorySpotlight",
        "isTimeshifted", "leadershipSkills", "originalReleaseDate", "otherFaceIds",
        "promoTypes", "rebalancedPrintings", "securityStamp", "side", "subsets", "watermark",
        "isFunny", "isAlternative", "isTextless", "keywords"
    ]:
        card.pop(key, None)

    return card


def json_walker(json_obj, callback, values_only=False):
    if isinstance(json_obj, dict):
        # If it's a dictionary, process each key-value pair
        modified_dict = {}
        for key, value in json_obj.items():
            if values_only is False:
                modified_key = callback(key)
            else:
                modified_key = key

            if isinstance(value, str):
                modified_value = callback(value)
            else:
                modified_value = json_walker(value, callback, values_only)

            modified_dict[modified_key] = modified_value
        return modified_dict
    elif isinstance(json_obj, list):
        # If it's a list, process each element
        return [json_walker(item, callback, values_only) for item in json_obj]
    elif isinstance(json_obj, str):
        # If it's a string, replace it with the result of the callback
        return callback(json_obj)
    else:
        # If it's any other type, return it as is
        return json_obj


def replace_card_name_with_tilde(text, card_name):
    # TODO: this doesn't quite work right for cards whose names end with a !, or for the card "+2 mace",
    #       because those either start or end with a word-break character, and so the card name won't cleanly
    #       match.
    return sub(r'\b%s\b' % escape(card_name), '~', text)


def replace_tilde_with_card_name(text, card_name):
    return sub(r'~', card_name, text)


def remove_reminder_text(text):
    return sub(r'\w\(.+\)', '', text)


def filter_empty_values(card):
    new_card = {}
    for key in card:
        if isinstance(card[key], bool) is False and len(card[key]) > 0:
            new_card[key] = card[key]

    return new_card


def replace_mdashes(text):
    return sub('--', '-', text)


def get_sequence_length_histogram(preprocessed_cards):
    card_lengths = {}
    for card in preprocessed_cards:
        if len(card) in card_lengths:
            card_lengths[len(card)] += 1
        else:
            card_lengths[len(card)] = 1
    return card_lengths


def get_token_frequencies(preprocessed_cards):
    token_frequencies = {}
    for card in preprocessed_cards:
        for token in card:
            if token in token_frequencies:
                token_frequencies[token] += 1
            else:
                token_frequencies[token] = 1
    return token_frequencies


def get_nth_percentile(data, percentile):
    # Flatten the frequency data
    flattened_data = []
    for length, freq in data.items():
        flattened_data.extend([length] * freq)

    # Sort the flattened data
    sorted_data = sorted(flattened_data)

    # Calculate the index for the nth percentile
    index = int(round(percentile / 100 * (len(sorted_data) - 1)))

    # Find and return the nth percentile value
    return sorted_data[index]


def main():
    parser = argparse.ArgumentParser(description='Prepare MTG JSON file for training')

    parser.add_argument('--input_path', action='store', type=str, default='corpus/AllPrintings.json',
                        help='Train the model using the specified data file')
    parser.add_argument('--output_path', action='store', type=str, default='corpus/preprocessed_cards.txt',
                        help='Where to save the pre-processed cards')

    args = parser.parse_args()

    # TODO: sanity checks on input_path and output_path (file/dir existence and/or create dir if needed)

    with open(args.input_path, 'r') as file:
        data = load(file)

    # TODO: make some of these filters into CLI args that can be toggled, for extra customization of the training set
    preprocessed_cards = []

    # Flatten and filter the card list
    for expansion in tqdm(data["data"].values()):
        for card in expansion["cards"]:
            # Skip card entries that we don't want in our training set
            if "side" in card and card["side"] != "a":
                continue

            if "modern" not in card["legalities"]:
                continue

            if card["legalities"]["modern"] != "Legal":
                continue

            if card["language"] != "English":
                continue

            if "name" not in card:
                continue

            # TODO: Don't include dual-face or split cards
            # TODO: Support dual-face cards
            # TODO: Support split cards
            # TODO: Support battles

            card = filter_unwanted_keys(card)
            card = filter_empty_values(card)
            card = json_walker(card, remove_reminder_text)
            card = json_walker(card, unidecode)
            card = json_walker(card, replace_mdashes)
            if "text" in card:
                card["text"] = replace_card_name_with_tilde(card["text"], card["name"])
                card["text"] = sub(r'\(.+?\)', '', card["text"])  # Strip reminder text
                card["text"] = card["text"].replace('\n', '\\n')  # Standardize the newline characters in the rules text

            preprocessed_cards.append(card)

    # Remove duplicates by name
    preprocessed_cards = {card["name"]: card for card in preprocessed_cards}.values()

    # Pre-tokenize the cards
    preprocessed_cards = [
        pre_tokenize_card(card)
        for card in preprocessed_cards
    ]

    # Filter any card that contains unique tokens
    # Identify tokens with keepable frequency
    token_frequencies = get_token_frequencies(preprocessed_cards)
    tokens_with_more_than_one_freq = {token for token, freq in token_frequencies.items() if freq > 1}

    # Filter out cards that contain any low-frequency token
    preprocessed_cards = [card for card in preprocessed_cards if
                      all(token in tokens_with_more_than_one_freq for token in card)]

    # Cards will have a long tail on maximum length, and training/inference can be sped up quite a bit by picking a
    # rational maximum card length (90th percentile)
    # TODO: make this cutoff configurable, CLI-passed
    stats = get_sequence_length_histogram(preprocessed_cards)
    cutoff_sequence_length = get_nth_percentile(stats, 90)
    preprocessed_cards = [
        card for card in preprocessed_cards if len(card) <= cutoff_sequence_length
    ]

    # Prepare the cards as ^-separated
    preprocessed_cards = [
        '^'.join(card)
        for card in preprocessed_cards
    ]

    # Save the output file
    with open(args.output_path, 'w', encoding='utf-8') as file:
        for card in preprocessed_cards:
            file.write(card)


def pre_tokenize_card(card):
    # Start the object
    tokens = ['{']

    # TODO: explode out the \w+|\W thing so that we can have a little more control on what counts as a word barrier
    #       (e.g. apostrophes are considered a word barrier right now, which isn't desirable)

    for key, value in card.items():
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
            tokens.append('"')
            # Let's let names be arbitrary constructions, instead of copied words
            if key == 'name':
                tokens.extend(list(value))
            else:
                # TODO: wrap the rules templates in _checks_ for \W on either side, but
                #       don't _capture_ the \W's
                regex = (
                        r'|'.join(rules_templates)  # rules templates
                        + r'|\\n'  # newlines, esp. in text
                        + r'|\{.+?}'  # mana symbols
                        + r'|\w+'  # whole words
                        + r'|\W'  # non-word characters
                )
                tokens.extend(findall(regex, value, re.IGNORECASE))
            tokens.append('",')

    # End the object
    last_token = tokens[-1]
    last_token = last_token[:-1]  # Remove the comma
    last_token += "}\n"  # Close the object
    tokens[-1] = last_token

    return tokens


if __name__ == "__main__":
    main()
