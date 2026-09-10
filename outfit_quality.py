"""Conservative metadata evidence shared by outfit and compatibility selection."""
STYLES = {
    'formal': {'formal', 'elegant', 'classic'},
    'sporty': {'sporty', 'athletic', 'activewear'},
    'casual': {'casual', 'streetwear', 'minimal', 'classic', 'bohemian'},
    'date': {'casual', 'elegant', 'classic'},
    'party': {'elegant', 'formal', 'streetwear', 'bold'},
}
KNOWN_STYLES = set().union(*STYLES.values())
NEUTRALS = {'black', 'white', 'grey', 'gray', 'navy', 'beige', 'brown', 'cream'}


def words(value):
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if isinstance(v, str)}
    return set(value.lower().replace(',', ' ').split()) if isinstance(value, str) else set()


def evidence(items, request):
    profile = request.get('style_profile') or {}
    profile = profile if isinstance(profile, dict) else {}
    avoid = words(profile.get('avoid')) - {'none'}
    preferred = words(profile.get('colors'))
    styles = STYLES.get(request.get('occasion'), set())
    vibe = profile.get('vibe')
    avoided = mismatch = season = preference = 0
    for item in items:
        colors = words(item.get('color')) | words(item.get('colors'))
        avoided += bool(colors & avoid)
        style = str(item.get('style', '')).lower()
        mismatch += bool(style in KNOWN_STYLES and styles and style not in styles)
        seasons = words(item.get('season'))
        season += bool((request.get('weather') == 'hot' and seasons == {'winter'}) or
                       (request.get('weather') == 'cold' and seasons == {'summer'}))
        preference += bool(style in KNOWN_STYLES and vibe in KNOWN_STYLES and style != vibe)
    # This is an internal ordering of evidence, not a consumer compatibility score.
    all_colors = set().union(*(words(i.get('color')) | words(i.get('colors')) for i in items))
    color_preference = int(bool(preferred and all_colors and not preferred.intersection(all_colors)))
    return avoided, mismatch, season, preference, color_preference


def explain(items, request):
    de = request.get('language') == 'de'
    cats = {i.get('category') for i in items}
    text = ('Das Kleid bildet die Basis dieses Looks.' if de else 'The dress forms the base of this look.') if 'dress' in cats else (
        'Oberteil und Unterteil bilden die Basis dieses Looks.' if de else 'The top and bottom form the base of this look.')
    colors = set().union(*(words(i.get('color')) | words(i.get('colors')) for i in items))
    if colors and colors <= NEUTRALS:
        text += ' Die erfassten Farben sind neutral.' if de else 'The recorded colours are neutral.'
    if request.get('weather') == 'cold' and 'outerwear' in cats:
        text += ' Für die Kälte ist eine äußere Schicht dabei.' if de else 'An outer layer is included for the cold setting.'
    if request.get('anchor_item_id'):
        text += ' Dein ausgewähltes Teil bleibt dabei.' if de else 'Your selected piece stays in the outfit.'
    uncertain = not all(i.get('style') and i.get('color') for i in items)
    if uncertain:
        text += ' Einige Angaben fehlen; prüfe Stil und Komfort selbst.' if de else 'Some details are missing; check the style and comfort yourself.'
    return text, ('Prüfe die Kombination an dir, bevor du sie einplanst.' if de else 'Try the combination on before planning to wear it.')
