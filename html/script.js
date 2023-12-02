import data from './card-data.js';

Handlebars.registerHelper('mana_symbols', (cost) => {
    const symbols = cost.split(/{(.+?)}/);
    return symbols.map((symbol) => symbol ? `<span class="symbol_${symbol}">${symbol}</span>` : '').join('');
})

Handlebars.registerHelper('card_background', (cost) => {
    const symbols = cost.split(/{(.+?)}/);
    return symbols.map((symbol) => symbol ? `background_${symbol}` : '').join(' ');
})

const source = document.getElementById("card-template").innerHTML;
const template = Handlebars.compile(source);

document.getElementById("card").innerHTML = template(data);
