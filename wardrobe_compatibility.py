"""Bounded, metadata-only purchase assessment. Never calls an AI service."""
from outfit_engine import build_batch, prepare, category, CATEGORIES
from outfit_quality import evidence, KNOWN_STYLES


def assess(request):
    new = request.get('new_item')
    if not isinstance(new, dict) or category(new) not in CATEGORIES:
        raise ValueError('A recognised clothing category is required.')
    owned, _ = prepare({'wardrobe': request.get('wardrobe', [])})
    anchor = 'potential-purchase'
    while any(i['id'] == anchor for i in owned):
        anchor += ':'
    new = {**new, 'id': anchor, 'category': category(new)}
    items = owned + [new]
    context = {**request, 'wardrobe': items, 'anchor_item_id': anchor, 'batch_size': 3}
    available = {category(i) for i in items}
    missing = []
    if new['category'] == 'top' and 'bottom' not in available:
        missing = ['bottom']
    elif new['category'] == 'bottom' and 'top' not in available:
        missing = ['top']
    elif 'dress' not in available and not {'top', 'bottom'} <= available:
        missing = [c for c in ['top', 'bottom'] if c not in available]
    result = build_batch(context)  # intentional local-only; bounded to engine candidate limit
    outfits = []
    supported = set()
    for look in result['outfits']:
        indices = [i['item_index'] for i in look['outfit']]
        selected = [items[i] for i in indices]
        # A best-of-a-weak-set result is not evidence for a good purchase.
        if any(evidence(selected, context)[:3]):
            continue
        enough = all(str(i.get('style', '')).lower() in KNOWN_STYLES
                     and str(i.get('color', '')).strip().lower() not in ('', 'unknown') for i in selected)
        owned_indices = [i for i in indices if i < len(owned)]
        if enough:
            supported.update(owned_indices)
        outfits.append({'wardrobe_indices': owned_indices, 'description': look['explanation'],
                        'suitability': 'metadata_supported' if enough else 'uncertain'})
    state = 'missing_essential' if missing else 'supported' if supported else 'uncertain'
    de = request.get('language') == 'de'
    messages = {
        'missing_essential': ('Für einen vollständigen Look fehlt eine wesentliche Kategorie. Das ist keine Aussage über Stil.',
                              'An essential category is missing for a complete look. This does not assess style.'),
        'supported': ('Diese Kombinationen sind durch die erfassten Angaben gestützt. Prüfe Passform und Wirkung selbst.',
                      'Recorded details support these combinations. Check their fit and appearance yourself.'),
        'uncertain': ('Die Stil-Eignung lässt sich aus den erfassten Angaben nicht sicher beurteilen. Das bedeutet nicht, dass nichts passt.',
                      'The recorded details do not establish suitability. This does not mean nothing matches.'),
    }
    return {'match_count': len(supported), 'matching_indices': sorted(supported), 'outfits': outfits,
            'verdict': messages[state][0 if de else 1], 'color_harmony': '', 'style_fit': '',
            'assessment_state': state, 'missing_categories': missing, 'complete_outfit_count': len(outfits),
            'search_bounded': True, 'assessment_source': 'local_metadata', 'compatibility_api_version': 1}
