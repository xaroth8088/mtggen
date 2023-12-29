# mtggen
 Experiments with using AI to generate novel Magic: the Gathering cards

_Obviously not affiliated with Wizards of the Coast, Hasbro, or any other rights holders_

# How to run
```bash
    conda env create -n mtggen -f environment.yml
    conda activate mtggen
    python main.py --help
```

# How to generate images for use in Tabletop Simulator
## Prereq's
### Chrome
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get update
sudo apt-get install -f
sudo apt-get install fonts-noto-color-emoji
```

### imagemagick
```bash
sudo apt-get install imagemagick
```

## Generating cards for Tabletop Simulator
The `generate_sheets.sh` script will create image files suitable for use in Tabletop Simulator.  Just tell it how many
cards you want, and it'll generate them and arrange them into 10x7 sheets of cards.  Keep in mind that it can easily
take 20-30 seconds to generate each card (or ~30 min. per sheet), so plan your day accordingly.

It's also important to note that a card may not always be fully generated, depending on how well your checkpoint was
trained. In these cases, you'll get a blank card instead of a generated one.

### One terminal for the webserver...
```bash
python main.py web mtggen.keras
```

### ...and another to make the sheets
```bash
./generate_sheets.sh 210
```
