import data from 'http://localhost:8000/card-data.js';

Handlebars.registerHelper('mana_symbols', (cost) => {
    return replaceSymbols(cost);
})

Handlebars.registerHelper('card_background', (cost) => {
    const symbols = cost.split(/{(.+?)}/);
    return symbols.map((symbol) => symbol ? `background_${symbol}` : '').join(' ');
})

Handlebars.registerHelper('rarity_symbol', (rarity) => {
    return `<span class="rarity_${rarity}">AI</span>`;
})

Handlebars.registerHelper('show_stats', (types) => {
    if(types.indexOf('Creature') > -1) {
        return "show_stats";
    }

    return "";
})

Handlebars.registerHelper('card_text', (text) => {
    if (text === undefined) {
        return text;
    }

    text = replaceSymbols(text);
    text = text.replaceAll('*', '&#x2022;')
    text = text.replaceAll('\n', '<div class="line_break"></div>');

    return text;
})

function replaceSymbols(text) {
    text = text.replaceAll(/{(.+?)}/g, `<span class="symbol symbol_$1">$1</span>`);
    return text;
}

const source = document.getElementById("card-template").innerHTML;
const template = Handlebars.compile(source);

document.getElementById("card").innerHTML = template(data);
