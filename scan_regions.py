"""Offline pair-level detection adapter. No model calls or manual annotations."""
import json
import math
import re
import cv2
import numpy as np

CATEGORIES = {'shoes','top','bottom','dress','outerwear','bag','jewelry','accessory'}


def components(removed):
    mask=(np.asarray(removed.convert('RGBA').getchannel('A'))>=64).astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    output={}
    for label in range(1,count):
        x,y,w,h,area=map(int,stats[label])
        if area>=.01*np.count_nonzero(mask):
            output[len(output)+1]={'bbox':[x,y,x+w,y+h],'mask':labels==label}
    return output


def box_valid(box,width,height):
    return (isinstance(box,list) and len(box)==4
        and all(type(v) in (int,float) and math.isfinite(v) for v in box)
        and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)


def parse(text):
    """Permit exactly one surrounding fence; validate everything inside it."""
    text=text.strip()
    fence=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',text,re.IGNORECASE)
    if fence:text=fence.group(1)
    value=json.loads(text,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value,dict) or not isinstance(value.get('items'),list):
        raise ValueError('Invalid detection object')
    return value


def adapt(text,removed):
    """Return records plus explicit diagnostics; invalid records are not hidden."""
    value=parse(text);width,height=removed.size;regions=components(removed)
    diagnostics=[]
    if type(value.get('complete')) is not bool or not isinstance(value.get('unresolved'),list):
        diagnostics.append('invalid_completeness')
    if value.get('complete') and value.get('unresolved'):diagnostics.append('contradictory_completeness')
    if not value.get('complete'):diagnostics.append('incomplete_detection')
    if len(value['items'])>4:
        # Do not truncate and silently lose detections. Caller can show the frame
        # failure for review; no candidates are sent into identity matching.
        return {'records':[],'diagnostics':diagnostics+['item_limit_exceeded'],'raw':value}
    owners={}
    for index,item in enumerate(value['items']):
        if isinstance(item,dict) and isinstance(item.get('component_ids'),list):
            for identity in item['component_ids']:
                if type(identity) is int:owners.setdefault(identity,[]).append(index)
    records=[];seen=set()
    for index,item in enumerate(value['items']):
        errors=[e for e in diagnostics if e!='incomplete_detection'];notes=[]
        if not isinstance(item,dict):
            records.append({'index':index,'raw':item,'diagnostics':['invalid_item'],'mask':None,'metadata':{}});continue
        identity=item.get('id')
        if not isinstance(identity,str) or not identity or identity in seen:errors.append('invalid_item_id')
        if isinstance(identity,str):seen.add(identity)
        if not isinstance(item.get('category'),str) or item['category'] not in CATEGORIES:errors.append('invalid_category')
        if not isinstance(item.get('subcategory'),str) or not item['subcategory'].strip():errors.append('missing_subtype')
        elif re.search(r'/|\bor\b',item['subcategory'],re.IGNORECASE):errors.append('ambiguous_subtype')
        if not isinstance(item.get('unit'),str) or item['unit'] not in {'pair','single'}:errors.append('invalid_unit')
        if not isinstance(item.get('confidence'),str) or item['confidence'] not in {'high','medium','low'}:errors.append('invalid_confidence')
        if not box_valid(item.get('bbox'),width,height):errors.append('invalid_coordinates')
        shoes=item.get('shoe_boxes',[]);visible=item.get('visible_shoes')
        # Per-shoe boxes refine a pair region; omission is not a count assertion.
        # Still validate every supplied box, including an incomplete list.
        if not isinstance(shoes,list) or not all(box_valid(b,width,height) for b in shoes):errors.append('invalid_shoe_coordinates')
        elif box_valid(item.get('bbox'),width,height) and any(
                b[0]<item['bbox'][0] or b[1]<item['bbox'][1] or b[2]>item['bbox'][2] or b[3]>item['bbox'][3] for b in shoes):
            errors.append('shoe_outside_pair')
        if item.get('category')=='shoes':
            if visible is None:
                errors.append('missing_visible_shoe_count')
            elif type(visible) is not int or visible<0 or visible>2 or item.get('unit')=='single' and visible>1:
                errors.append('shoe_count_contradiction')
            elif type(visible) is int and isinstance(shoes,list) and len(shoes)>visible:
                errors.append('shoe_count_contradiction')
            elif isinstance(shoes,list) and (not shoes or visible is None or len(shoes)<visible):
                notes.append('incomplete_optional_shoe_details')
            if item.get('unit')!='pair' or type(visible) is int and visible<2:errors.append('partial_pair')
            if visible is None:notes.append('visible_shoe_count_unspecified')
        elif shoes or visible not in (None,0):errors.append('unexpected_shoe_details')
        ids=item.get('component_ids')
        valid_ids=isinstance(ids,list) and all(type(i) is int and i in regions for i in ids)
        if not valid_ids:errors.append('invalid_components')
        elif any(len(owners[i])>1 for i in ids):errors.append('conflicting_components')
        mask=None
        if valid_ids and ids and not any(e in errors for e in ('invalid_coordinates','conflicting_components')):
            box=item['bbox']
            if any(not (box[0]<=regions[i]['bbox'][0] and box[1]<=regions[i]['bbox'][1]
                        and box[2]>=regions[i]['bbox'][2] and box[3]>=regions[i]['bbox'][3]) for i in ids):
                errors.append('region_component_conflict')
            else:mask=np.logical_or.reduce([regions[i]['mask'] for i in ids])
        if mask is None:errors.extend(['missing_segmentation','unverified_source_region'])
        # No manual normalization or invented color/fabric/style metadata.
        metadata={k:item.get(k) for k in ('category','subcategory','confidence')}
        for key in metadata:
            if not isinstance(metadata[key],str):metadata[key]=None
        if 'invalid_category' in errors:metadata['category']=None
        records.append({'index':index,'raw':item,'metadata':metadata,'mask':mask,
                        'diagnostics':list(dict.fromkeys(errors)),'notes':notes})
    return {'records':records,'diagnostics':diagnostics,'raw':value}
