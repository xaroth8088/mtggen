# Add any common phrases that appear in the rules here.  These will be tokenized, so don't try to get fancy with them.
# Ideally, these should be very common phrases, appearing at least 500 times in the modern cards corpus.
# It's also best if they are fully-formed, representing a single game instruction or key concept
# Because there can be a lot of overlap in phrasing, they should also be roughly ordered by descending frequency,
# with more specific phrasings taking precedence over less specific phrasings.
rules_templates = [
    r"can't",   # TODO: once we have fully custom word boundaries in place, we can remove this
    r"creature's",   # TODO: once we have fully custom word boundaries in place, we can remove this
    r"~'s",   # TODO: once we have fully custom word boundaries in place, we can remove this
    r'\\\\n',
    r'you control', # 4729
    r'target creature', # 3749
    r'you may', # 3721
    r'until end of turn', # 3390
    r'when ~ enters the battlefield', # 2265
    r'[^ ]+ counters?',
    r'[-+][0-9]+/[-+][0-9]+', # TODO: Maybe the token should be '+x', instead of '+x/+x'
    r'your graveyard', # 1886
    r'your hand', # 1886
    r'draw \w+ cards?', # 1886
    r'this creature', #1752
    r'deals [0-9x]+ damage', #1389
    r'creature tokens?', #1364
    r'creature you control', # 1340
    r'discard .*? cards?', # 860
    r'as long as', #823
    r'that creature', #796
    r'1/1', # 774
    r'enchanted creature', # 782
    r'combat damage', # 736
    r'under your control', #716
    r'if you do,', # 672
    r'sacrifice ~', #669
    r'target player', # 654
    r'can\'t be blocked', # 653
    r'where x is the number of', #248
    r'is the number of', #607 - 248
    r'search your library', # 603
    r'each creature', #587
    r'onto the battlefield', # 553
    r'at the beginning of your upkeep,', # 531
    r'any target', # 525
    r'only as a sorcery.', # 506
    r"owner's hand", #497
    r'~ attacks', #485
    r'first strike', #444
    r'the top card of your library', # 354
    r'scry [0-9]+', #334
    r'add one mana', # 304
    r'your opponents control',  # 243
    r'your opponents',  # 327 minus the 243 from 'your opponents control', above
    r'bottom of your library', #257
    r'casts? a spell', #221 (50 for 'casts', 171 for 'cast')
    r'target spell', #217
    r'at the beginning of the next end step', #215
    r'gain life', #197
    r'activate only as a sorcery', #193
    r'double strike', #173
    r'mill .*? cards?', #158
]
