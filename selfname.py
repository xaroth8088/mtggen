import json
import sys
import re

# Read JSON data from stdin
input_data = json.load(sys.stdin)

# Define a function to replace occurrences of ".name" with "~"
def replace_name_with_self(obj):
    if isinstance(obj, list):
        return [replace_name_with_self(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: re.sub(r'\b%s\b' % re.escape(obj['name']), '~', value) if key == 'text' else value for key, value in obj.items()}
    else:
        return obj

# Use the function to modify the JSON data
modified_data = replace_name_with_self(input_data)

# TODO: translate the jq bits into Python code
# TODO: replace special characters with their ASCII equivalents.  e.g. emdash with hyphen, accented characters, bullets to asterisks, etc.
# TODO: end each line with a special character like "@", so that we can know for sure where to trim the candidate generations

# Print the modified JSON data
json.dump(modified_data, sys.stdout, ensure_ascii=False, indent=2)

