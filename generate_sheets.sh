#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_times>"
    exit 1
fi

n=$1
counter=1

# Create directories for screenshots and sheets
mkdir -p screenshots
mkdir -p sheets

# Run the Chrome command n times
while [ $counter -le $n ]
do
    google-chrome --headless --disable-gpu --screenshot=screenshots/out_$counter.png --no-sandbox --window-size=260,355 http://localhost:8000/html/card.html
    let counter=counter+1
done

# Check if ImageMagick is installed
if ! command -v montage &> /dev/null
then
    echo "ImageMagick is not installed. Please install it to create image sheets."
    exit 1
fi

# Create sheets of 10x7 images
ls screenshots/*.png | xargs -n 70 | while read line; do
    montage $line -tile 10x7 -geometry +0+0 sheets/sheet_$((++sheet_counter)).png
done

echo "Process completed. Check the screenshots and sheets directories."
