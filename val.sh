#!/bin/bash

# This script validates that all lines in the file are valid for use with mtggen.
# Each line has to meet these criteria:
# 1. it must be a valid JSON object
# 2. values must be either a) a string, or b) an array of strings

line_number=0
while IFS= read -r line; do
    ((line_number++))

    if [ $((line_number % 100)) -eq 0 ]; then
      echo "$line_number"
    fi

    # Validate JSON using jq
    validation_result=$(echo "$line" | jq 'all(.[]; type == "string" or (type == "array" and all(.[]; type == "string")))')

    # Check validation result
    if [ "$validation_result" != "true" ]; then
        echo "JSON validation failed: Some keys have values that are not strings or arrays of strings."
        echo "$line"
        exit 1
    fi
done < filtered.jsonl

echo "All lines in the JSONL file passed validation."
exit 0
