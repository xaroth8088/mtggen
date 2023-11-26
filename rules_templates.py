# Add any common phrases that appear in the rules here.  These will be tokenized, so don't try to get fancy with them.
# Ideally, these should be very common phrases, appearing at least 500 times in the modern cards corpus.
# It's also best if they are fully-formed, representing a single game instruction or key concept
# Because there can be a lot of overlap in phrasing, they should also be roughly ordered by descending frequency,
# with more specific phrasings taking precedence over less specific phrasings.
rules_templates = [
    r'you control', # 4729
    r'target creature', # 3749
    r'you may', # 3721
    r'until end of turn', # 3390
    r'when ~ enters the battlefield', # 2265
    r'\+1/\+1 counter', # 2010
    r'\+1/\+1', # 2948, minus the 2010 from above
    r'your graveyard', # 1886
    r'your hand', # 1886
    r'draw \w+ cards?', # 1886
    r'creature you control', # 1340
    r'1/1', # 774
    r'enchanted creature', # 782
    r'combat damage', # 736
    r'if you do,', # 672
    r'target player', # 654
    r'can\'t be blocked', # 653
    r'search your library', # 603
    r'you may pay',  # 556
    r'onto the battlefield', # 553
    r'at the beginning of your upkeep,', # 531
    r'any target', # 525
    r'only as a sorcery.', # 506
    r'discard a card', # 416
    r'the top card of your library', # 354
    r'add one mana', # 304
    r'-1/-1 counter',  # 292
    r'-1/-1',  # 420, minus the 292 from above
    r'your opponents control',  # 243
    r'your opponents',  # 327 minus the 243 from 'your opponents control', above
]
