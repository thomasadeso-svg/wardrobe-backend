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
    labels = {'top': 'Oberteil', 'bottom': 'Unterteil', 'dress': 'Kleid', 'shoes': 'Schuhe',
              'outerwear': 'Jacke/Mantel', 'bag': 'Tasche', 'jewelry': 'Schmuck', 'accessory': 'Accessoire'}
    details = []
    for item in items:
        name = item.get('name')
        label = name.strip() if isinstance(name, str) and name.strip() else (
            labels.get(item.get('category'), 'Teil') if de else item.get('subcategory') or item.get('category') or 'item')
        attributes = [item.get(key).strip() for key in ('color', 'style')
                      if isinstance(item.get(key), str) and item[key].strip().lower() not in ('', 'unknown')]
        if attributes:
            details.append(f"{label} ({', '.join(attributes)})")
    if details:
        text += (' Erfasste Details: ' if de else 'Recorded details: ') + '; '.join(details) + '.'
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
    tip = ('Trage die äußere Schicht offen, um die Kleidung darunter zu zeigen.' if de else
           'Wear the outer layer open to show the pieces underneath.') if 'outerwear' in cats else (
           'Prüfe beim Anprobieren die Länge des Oberteils zum Unterteil.' if de else
           'When trying these on, check where the top hem sits against the bottom.') if 'top' in cats else (
           'Prüfe beim Anprobieren, wie der Kleidsaum fällt.' if de else 'When trying this on, check how the dress hem falls.')
    return text, tip


def match_info(items, request, wardrobe, exhaustive=True, generation_error=False):
    """Conservative named states from recorded metadata, never novelty or source."""
    de=request.get('language')=='de'
    core=[i for i in items if i.get('category') in {'top','bottom','dress','shoes','outerwear'}]
    unknown_categories=any(i.get('category') not in {'top','bottom','dress','shoes','outerwear','bag','jewelry','accessory'} for i in wardrobe)
    codes=[];alternatives=[]
    if not items:
        cats={i.get('category') for i in wardrobe}
        anchor=next((i for i in wardrobe if i.get('id')==request.get('anchor_item_id')),None)
        anchor_cat=anchor.get('category') if anchor else None
        if unknown_categories or not exhaustive:
            state='uncertain';codes=['missing_metadata' if unknown_categories else 'limited_search']
        else:
            state='no_sensible';codes=['missing_core']
            if anchor_cat=='top':alternatives=['bottom']
            elif anchor_cat=='bottom':alternatives=['top']
            elif 'top' in cats and 'bottom' not in cats:alternatives=['bottom']
            elif 'bottom' in cats and 'top' not in cats:alternatives=['top']
            else:alternatives=['dress']
    else:
        score=evidence(items,request)
        profile=request.get('style_profile') or {}
        profile=profile if isinstance(profile,dict) else {}
        needs_color=bool(words(profile.get('avoid'))-{'none'} or words(profile.get('colors')))
        uncertain=unknown_categories or any(
            str(i.get('style') or '').lower() not in KNOWN_STYLES
            or (request.get('weather') in {'hot','cold'} and not (words(i.get('season'))-{'unknown',''})) for i in core)
        uncertain=uncertain or (needs_color and any(not ((words(i.get('color')) | words(i.get('colors')))-{'unknown',''}) for i in items))
        # Quiz colour families are not literal hues. Without recorded family/shade
        # evidence, do not claim e.g. "pastels are absent" from just "pink".
        preferred=words(profile.get('colors'))
        recorded=set().union(*(words(i.get('color')) | words(i.get('colors')) for i in items))
        uncertain=uncertain or bool(preferred & {'earth','pastels','bright'} and not preferred.intersection(recorded))
        # Explicit avoid preferences and exclusive opposite-season metadata are
        # requirements; ordinary aesthetic differences are not hard exclusions.
        athletic_conflict=any((request.get('occasion')=='formal' and str(i.get('style','')).lower() in {'sporty','athletic','activewear'})
            or (request.get('occasion')=='sporty' and str(i.get('style','')).lower() in {'formal','elegant'}) for i in core)
        hard=bool(score[0] or score[2] or athletic_conflict)
        if hard:
            state='no_sensible'
            codes=(['avoid_conflict'] if score[0] else [])+(['season_conflict'] if score[2] else [])+(['activity_conflict'] if athletic_conflict else [])
            if uncertain:codes.append('missing_metadata')
        elif uncertain:state='uncertain';codes=['missing_metadata']
        elif any(score[1:]):
            state='best_available'
            codes=(['occasion_difference'] if score[1] else [])+(['style_preference'] if score[3] else [])+(['color_preference'] if score[4] else [])
        else:state='strong';codes=['supported_owned']
        if state in {'best_available','no_sensible'}:
            alternatives=list(dict.fromkeys(i['category'] for i in core if i['category'] in {'top','bottom','dress'}
                and any(evidence([i],request))))[:3]
    texts={
      'supported_owned':('Recorded details support this owned outfit for your selection.','Die erfassten Details unterstützen dieses eigene Outfit für deine Auswahl.'),
      'missing_metadata':('Relevant garment details are missing or cannot be assessed reliably. Suitability is uncertain; review your wardrobe details.','Relevante Kleidungsangaben fehlen oder lassen sich nicht zuverlässig beurteilen. Die Eignung ist ungewiss; prüfe deine Kleiderschrank-Angaben.'),
      'limited_search':('The checked combinations did not establish a suitable outfit; the owned search was limited.','Die geprüften Kombinationen ergaben kein gesichert geeignetes Outfit; die Suche in eigenen Teilen war begrenzt.'),
      'missing_core':('The owned selection is missing a dress or the required top-and-bottom combination.','Der eigenen Auswahl fehlt ein Kleid oder die erforderliche Kombination aus Oberteil und Unterteil.'),
      'avoid_conflict':('The checked option includes a recorded colour you asked to avoid.','Die geprüfte Option enthält eine erfasste Farbe, die du vermeiden möchtest.'),
      'season_conflict':('The available option includes a piece recorded only for the opposite temperature setting.','Die verfügbare Option enthält ein Teil, das nur für die gegensätzliche Temperatur angegeben ist.'),
      'activity_conflict':('Recorded athletic and formal clothing requirements conflict for this occasion.','Die erfasste Sport- bzw. Festkleidung passt nicht zu den Anforderungen dieses Anlasses.'),
      'occasion_difference':('This is a complete owned option; some recorded styles differ from the selected occasion.','Dies ist eine vollständige eigene Option; einige erfasste Stile weichen vom gewählten Anlass ab.'),
      'style_preference':('This owned outfit differs from your recorded preferred style.','Dieses eigene Outfit weicht von deinem angegebenen bevorzugten Stil ab.'),
      'color_preference':('Your preferred colours are not represented in the recorded colours of this outfit.','Deine bevorzugten Farben sind in den erfassten Farben dieses Outfits nicht vertreten.'),
    }
    return {'state':state,'reason_codes':codes,'reason': ' '.join(texts[c][int(de)] for c in codes),
            'alternative_categories':alternatives if exhaustive and not generation_error and 'missing_metadata' not in codes and state in {'best_available','no_sensible'} else [],
            'owned_search_complete':exhaustive,'generation_error':generation_error}
