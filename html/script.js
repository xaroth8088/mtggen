import data from 'http://localhost:8000/card-data.js';
//import data from './sample.js';

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
    if (types.indexOf('Creature') > -1) {
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

Handlebars.registerHelper('card_image', (name, types, subtypes, supertypes) => {
    const url = new URL('../../card-image.png', document.baseURI);
    const params = new URLSearchParams();
    params.append("name", name);
    if (types !== undefined) {
        types.forEach(type => params.append("type[]", type));
    }
    if (subtypes !== undefined) {
        subtypes.forEach(subtype => params.append("subtype[]", subtype));
    }
    if (supertypes !== undefined) {
        supertypes.forEach(supertype => params.append("supertype[]", supertype));
    }
    url.search = params;
    return url.toString()
})

function replaceSymbols(text) {
    return text
        .replace(/\{W\}/g, '<span class="symbol symbol_W"><span class="emoji">☀</span></span>')
        .replace(/\{U\}/g, '<span class="symbol symbol_U"><span class="emoji">💧</span></span>')
        .replace(/\{B\}/g, '<span class="symbol symbol_B"><span class="emoji">💀</span></span>')
        .replace(/\{R\}/g, '<span class="symbol symbol_R"><span class="emoji">🔥</span></span>')
        .replace(/\{G\}/g, '<span class="symbol symbol_G"><span class="emoji">🌳</span></span>')
        .replace(/\{S\}/g, '<span class="symbol symbol_S"><span class="emoji">❄️</span></span>')
        .replace(/\{T\}/g, '<span class="symbol symbol_tap">↷</span>')
        .replace(/\{\d+\}/g, match => `<span class="symbol">${match.slice(1, -1)}</span>`)
        .replaceAll(/{(.+?)}/g, `<span class="symbol symbol_$1">$1</span>`);
}

function resizeText(selector) {
    const container = document.body.querySelector(selector);
    const currentFontSize = parseFloat(
        getComputedStyle(container).fontSize.replace('px', '')
    );
    textFit(container, {minFontSize: 1, maxFontSize: currentFontSize});
}

const source = document.getElementById("card-template").innerHTML;
const template = Handlebars.compile(source);

document.getElementById("card").innerHTML = template(data);
resizeText('#card .name')
resizeText('#card .type')
resizeText('#card .text')
