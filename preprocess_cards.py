import argparse
import re
from json import load, dumps

from tqdm import tqdm
from unidecode import unidecode


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
        "isFunny", "isAlternative", "isTextless"
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
    return re.sub(r'\b%s\b' % re.escape(card_name), '~', text)


def remove_reminder_text(text):
    return re.sub(r'\w\(.+\)', '', text)


def filter_empty_values(card):
    new_card = {}
    for key in card:
        if isinstance(card[key], bool) is False and len(card[key]) > 0:
            new_card[key] = card[key]

    return new_card


def replace_mdashes(text):
    return re.sub('--', '-', text)


def main():
    parser = argparse.ArgumentParser(description='Prepare MTG JSON file for training')

    parser.add_argument('--input_path', action='store', type=str, default='corpus/AllPrintings.json',
                        help='Train the model using the specified data file')
    parser.add_argument('--output_path', action='store', type=str, default='corpus/preprocessed_cards.json',
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

            card = filter_unwanted_keys(card)
            card = filter_empty_values(card)
            card = json_walker(card, remove_reminder_text)
            card = json_walker(card, unidecode)
            card = json_walker(card, replace_mdashes)
            if "text" in card:
                card["text"] = replace_card_name_with_tilde(card["text"], card["name"])

            preprocessed_cards.append(card)

    # Remove duplicates by name
    preprocessed_cards = {card["name"]: card for card in preprocessed_cards}.values()

    # Save the output file
    with open(args.output_path, 'w', encoding='utf-8') as file:
        for card in preprocessed_cards:
            file.write(dumps(card))
            file.write('\n')


if __name__ == "__main__":
    main()
