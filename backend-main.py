from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import anthropic
import os
import io
import base64
import json
import random
import re
import hashlib
import requests
from PIL import Image, ImageEnhance
from rembg import remove, new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Initialize rembg session ONCE at startup
rembg_session = new_session("u2netp")

# Simple in-memory cache for vacation list results (resets on Railway redeploy)
vacation_cache = {}

# Allowed activities for v1
VALID_ACTIVITIES = {"beach", "dinner", "sightseeing", "nightlife", "hiking", "business", "casual", "workout"}


@app.get("/")
async def root():
    return {
        "status": "live",
        "service": "styligma-v2",
        "endpoints": [
            "/remove-background",
            "/analyze-clothing",
            "/generate-outfit",
            "/match-item",
            "/vacation-list",
            "/privacy",
            "/terms",
        ],
        "rembg": True,
        "remove_bg": bool(REMOVE_BG_API_KEY),
        "claude": bool(ANTHROPIC_API_KEY),
        "vacation_cache_size": len(vacation_cache),
    }


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        input_data = await file.read()

        img_input = Image.open(io.BytesIO(input_data))
        max_size = 1024
        if max(img_input.size) > max_size:
            img_input.thumbnail((max_size, max_size), Image.LANCZOS)
            buf_resized = io.BytesIO()
            img_input.save(buf_resized, format="JPEG", quality=90)
            input_data = buf_resized.getvalue()

        output_data = remove(input_data, session=rembg_session)

        img = Image.open(io.BytesIO(output_data)).convert("RGBA")
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))

        enhancer = ImageEnhance.Color(rgb_img)
        rgb_img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Contrast(rgb_img)
        rgb_img = enhancer.enhance(1.1)

        r2, g2, b2 = rgb_img.split()
        img = Image.merge("RGBA", (r2, g2, b2, a))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "success": True,
            "image": f"data:image/png;base64,{base64_image}",
            "method": "rembg-u2netp",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": file.content_type or "image/jpeg", "data": base64_image},
                    },
                    {
                        "type": "text",
                        "text": """You are a strict fashion wardrobe gatekeeper. Your ONLY job is to accept real clothing items and fashion accessories, and REJECT everything else.

STEP 1 — Is this a wearable fashion item?

ACCEPTED (return rejected=false):
Shirts, t-shirts, blouses, sweaters, hoodies, jackets, coats, blazers, vests, pants, jeans, shorts, skirts, dresses, shoes, sneakers, boots, sandals, heels, bags, handbags, backpacks, belts, watches, necklaces, bracelets, rings, earrings, sunglasses, scarves, hats, caps, ties, gloves, socks.

REJECTED (return rejected=true):
People, selfies, faces, body parts, food, drinks, animals, pets, cars, vehicles, furniture, rooms, buildings, landscapes, electronics, phones, laptops, books, plants, flowers, toys, tools, money, cups, mugs, plates, or ANY object that is not worn on the body. Also reject blurry or unrecognizable images.

When in doubt, REJECT.

If REJECTED, return ONLY this JSON:
{"rejected": true, "reason": "This doesn't look like a clothing item or accessory. Please photograph a single piece of clothing, shoes, bag, jewelry, or accessory."}

STEP 2 — If ACCEPTED, return ONLY this JSON:
{
  "rejected": false,
  "category": "top" or "bottom" or "shoes" or "outerwear" or "bag" or "jewelry" or "accessory",
  "subcategory": "e.g. t-shirt, jeans, sneakers, jacket, hat, necklace, handbag, sunglasses, belt, watch, scarf",
  "color": "primary color name",
  "colors": ["primary", "secondary if any"],
  "style": "casual" or "formal" or "sporty" or "streetwear" or "elegant" or "bohemian",
  "season": ["spring", "summer", "fall", "winter"],
  "fabric_guess": "e.g. cotton, denim, leather, polyester, gold, silver, canvas",
  "name": "Short descriptive name like 'Black Slim Jeans' or 'Gold Chain Necklace'"
}

Return ONLY the JSON, no other text.""",
                    },
                ],
            }],
        )

        response_text = message.content[0].text.strip()

        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            raise ValueError("Claude didn't return valid JSON")

        result = json.loads(match.group(0))
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        return JSONResponse(content={"rejected": True, "reason": "Could not analyze this image. Please try again with a clear photo of a clothing item."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    wardrobe = request.get("wardrobe", [])
    occasion = request.get("occasion", "casual")
    weather = request.get("weather", "moderate")
    previous_outfits = request.get("previous_outfits", [])
    style_profile = request.get("style_profile", None)

    if len(wardrobe) < 2:
        return {
            "outfit": [],
            "explanation": "Add more items! You need at least one top and one bottom.",
            "styling_tip": "",
        }

    items_list = []
    for i, item in enumerate(wardrobe):
        items_list.append(
            f"[{i}] {item.get('name', 'Item')} — {item.get('category', '?')}, "
            f"Color: {item.get('color', '?')}, Style: {item.get('style', '?')}, "
            f"Sub: {item.get('subcategory', '?')}"
        )
    items_text = "\n".join(items_list)

    avoid_text = ""
    if previous_outfits:
        avoid_lines = []
        for prev in previous_outfits[-5:]:
            avoid_lines.append(str(prev))
        avoid_text = f"\n\nAVOID these combinations (already shown):\n" + "\n".join(avoid_lines)

    profile_text = ""
    if style_profile:
        parts = []
        if style_profile.get("vibe"):
            parts.append(f"Style vibe: {style_profile['vibe']}")
        if style_profile.get("colors"):
            parts.append(f"Preferred colors: {', '.join(style_profile['colors'])}")
        if style_profile.get("avoid"):
            avoid_colors = [c for c in style_profile['avoid'] if c != 'none']
            if avoid_colors:
                parts.append(f"Colors to AVOID: {', '.join(avoid_colors)}")
        if style_profile.get("bodyFocus"):
            parts.append(f"Body focus: {style_profile['bodyFocus']}")
        if parts:
            profile_text = "\n\nUSER STYLE PROFILE (personalize to match):\n" + "\n".join(f"- {p}" for p in parts)

    if client:
        try:
            pick_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""You are an expert fashion stylist for "Styligma ✧".

WARDROBE:
{items_text}
{avoid_text}
{profile_text}

Pick the BEST outfit for:
- Occasion: {occasion}
- Weather: {weather}

STRICT RULES — MUST FOLLOW:
1. Pick EXACTLY 1 top (REQUIRED)
2. Pick EXACTLY 1 bottom (REQUIRED)
3. Pick EXACTLY 1 shoes (if available)
4. If weather is "cold": Pick EXACTLY 1 outerwear (REQUIRED)
5. If weather is "moderate": Outerwear optional (0 or 1)
6. If weather is "hot": NO outerwear
7. Optionally 1 bag/jewelry/accessory (max 1)

ABSOLUTE RULES:
- NEVER pick 2 items from the same category
- NEVER pick 2 tops, 2 bottoms, 2 shoes, or 2 outerwear
- Each category appears AT MOST ONCE
- Total items: 3-5, never more

VARIETY RULES:
- DO NOT repeat previous combinations
- Rotate through available items

STYLE RULES:
- Focus on COLOR HARMONY: complementary, analogous, or monochrome palettes
- Match STYLE: don't mix sporty with elegant unless streetwear
- Consider fabric/texture combos
- Be creative — surprise with unexpected but fashionable pairings

Return ONLY JSON:
{{
  "selected_indices": [0, 3, 5],
  "explanation": "Why these pieces work — mention specific colors and textures",
  "styling_tip": "One specific actionable tip for wearing this outfit"
}}

selected_indices = exact index numbers from the list. ONLY JSON, nothing else.""",
                }],
            )

            pick_text = pick_response.content[0].text.strip()

            match = re.search(r'\{.*\}', pick_text, re.DOTALL)
            if not match:
                raise ValueError("Claude didn't return valid JSON")

            pick_data = json.loads(match.group(0))
            indices = pick_data.get("selected_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(wardrobe)]

            seen_categories = set()
            deduplicated = []
            for i in valid_indices:
                cat = wardrobe[i].get("category", "unknown").lower().strip()
                if cat not in seen_categories:
                    seen_categories.add(cat)
                    deduplicated.append(i)

            category_order = {"outerwear": 0, "top": 1, "bottom": 2, "shoes": 3, "bag": 4, "jewelry": 5, "accessory": 6}
            deduplicated.sort(key=lambda x: category_order.get(wardrobe[x].get("category", "unknown").lower().strip(), 99))

            valid_indices = deduplicated

            if valid_indices:
                return {
                    "outfit": [{"item_index": i} for i in valid_indices],
                    "explanation": pick_data.get("explanation", "A curated look styled by AI."),
                    "styling_tip": pick_data.get("styling_tip", "Own it with confidence."),
                }
        except Exception as e:
            print(f"AI outfit selection failed: {e}")

    tops = [i for i, item in enumerate(wardrobe) if item.get("category", "").lower() == "top"]
    bottoms = [i for i, item in enumerate(wardrobe) if item.get("category", "").lower() == "bottom"]
    shoes = [i for i, item in enumerate(wardrobe) if item.get("category", "").lower() == "shoes"]
    outerwear = [i for i, item in enumerate(wardrobe) if item.get("category", "").lower() == "outerwear"]

    if not tops or not bottoms:
        return {"outfit": [], "explanation": "Need at least one top and one bottom.", "styling_tip": ""}

    for lst in [tops, bottoms, shoes, outerwear]:
        random.shuffle(lst)

    selected = [{"item_index": tops[0]}, {"item_index": bottoms[0]}]
    if shoes:
        selected.append({"item_index": shoes[0]})
    if outerwear and weather in ["cold", "moderate"]:
        selected.insert(0, {"item_index": outerwear[0]})

    return {
        "outfit": selected,
        "explanation": "A fresh combination from your wardrobe.",
        "styling_tip": "Mix textures for a balanced silhouette.",
    }


@app.post("/match-item")
async def match_item(request: dict):
    new_item = request.get("new_item", {})
    wardrobe = request.get("wardrobe", [])
    occasion = request.get("occasion", "casual")

    if not new_item or len(wardrobe) < 1:
        return {
            "matches": [],
            "outfits": [],
            "verdict": "Add more items to your wardrobe first.",
            "match_count": 0,
        }

    items_list = []
    for i, item in enumerate(wardrobe):
        items_list.append(
            f"[{i}] {item.get('name', 'Item')} — {item.get('category', '?')}, "
            f"Color: {item.get('color', '?')}, Style: {item.get('style', '?')}, "
            f"Sub: {item.get('subcategory', '?')}"
        )
    items_text = "\n".join(items_list)

    new_item_desc = (
        f"{new_item.get('name', 'Item')} — {new_item.get('category', '?')}, "
        f"Color: {new_item.get('color', '?')}, Style: {new_item.get('style', '?')}, "
        f"Sub: {new_item.get('subcategory', '?')}"
    )

    if client:
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": f"""You are an expert fashion stylist for "Styligma ✧".

A user is considering BUYING this new item:
NEW ITEM: {new_item_desc}

Their current WARDROBE:
{items_text}

TASK: Determine how well this new item fits into their existing wardrobe.

1. Find ALL wardrobe items that would pair well with this new item
2. Suggest up to 3 complete outfits using the new item + wardrobe items
3. Give a verdict: is this a SMART BUY or redundant?

Return ONLY JSON:
{{
  "match_count": <number of items that pair well>,
  "matching_indices": [list of wardrobe indices that pair with the new item],
  "outfits": [
    {{
      "wardrobe_indices": [indices from wardrobe to combine with new item],
      "description": "Short outfit description"
    }}
  ],
  "verdict": "SMART BUY: <reason>" or "SKIP: <reason>" or "MAYBE: <reason>",
  "color_harmony": "Brief note on how the new item's color works with wardrobe",
  "style_fit": "How well it matches the user's overall style"
}}

Be honest — if the user already has something similar, say SKIP. If it fills a gap, say SMART BUY.
ONLY JSON, nothing else.""",
                }],
            )

            pick_text = response.content[0].text.strip()

            match = re.search(r'\{.*\}', pick_text, re.DOTALL)
            if not match:
                raise ValueError("Claude didn't return valid JSON")

            result = json.loads(match.group(0))

            valid_matches = [i for i in result.get("matching_indices", []) if 0 <= i < len(wardrobe)]
            valid_outfits = []
            for outfit in result.get("outfits", [])[:3]:
                valid_idx = [i for i in outfit.get("wardrobe_indices", []) if 0 <= i < len(wardrobe)]
                if valid_idx:
                    valid_outfits.append({
                        "wardrobe_indices": valid_idx,
                        "description": outfit.get("description", ""),
                    })

            return {
                "match_count": len(valid_matches),
                "matching_indices": valid_matches,
                "outfits": valid_outfits,
                "verdict": result.get("verdict", "Looks like a versatile addition."),
                "color_harmony": result.get("color_harmony", ""),
                "style_fit": result.get("style_fit", ""),
            }
        except Exception as e:
            print(f"Match item failed: {e}")

    new_cat = new_item.get("category", "")
    matches = []
    for i, item in enumerate(wardrobe):
        cat = item.get("category", "")
        if cat != new_cat:
            matches.append(i)

    return {
        "match_count": len(matches),
        "matching_indices": matches[:10],
        "outfits": [],
        "verdict": f"This pairs with {len(matches)} items in your wardrobe.",
        "color_harmony": "",
        "style_fit": "",
    }


@app.post("/vacation-list")
async def vacation_list(request: dict):
    """
    Generate a vacation packing list + daily outfits from the user's wardrobe.

    Request body:
    {
      "wardrobe": [...],
      "destination": "Barcelona",
      "start_date": "2026-06-14",
      "end_date": "2026-06-21",
      "activities": ["beach", "dinner", "sightseeing", "casual"],
      "weather": "warm",
      "style_profile": {...}
    }
    """
    wardrobe = request.get("wardrobe", [])
    destination = request.get("destination", "").strip()
    start_date = request.get("start_date", "").strip()
    end_date = request.get("end_date", "").strip()
    activities = request.get("activities", [])
    weather = request.get("weather", "moderate")
    style_profile = request.get("style_profile", None)

    if len(wardrobe) < 5:
        return JSONResponse(
            status_code=400,
            content={"error": "Wardrobe needs at least 5 items to plan a trip."},
        )
    if not destination:
        return JSONResponse(status_code=400, content={"error": "Destination is required."})
    if not start_date or not end_date:
        return JSONResponse(status_code=400, content={"error": "Start and end dates are required."})

    try:
        from datetime import date
        d1 = date.fromisoformat(start_date)
        d2 = date.fromisoformat(end_date)
        days = (d2 - d1).days + 1
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid date format. Use YYYY-MM-DD."})

    if days < 1:
        return JSONResponse(status_code=400, content={"error": "End date must be after start date."})
    if days > 14:
        return JSONResponse(status_code=400, content={"error": "Trips longer than 14 days aren't supported yet."})

    normalized_activities = [a.lower().strip() for a in activities if isinstance(a, str)]
    normalized_activities = [a for a in normalized_activities if a in VALID_ACTIVITIES]
    if not normalized_activities:
        normalized_activities = ["casual"]

    # Cache key based on wardrobe signature + trip details
    wardrobe_signature = "|".join(
        f"{item.get('name', '')}-{item.get('category', '')}-{item.get('color', '')}"
        for item in wardrobe
    )
    cache_payload = json.dumps({
        "wardrobe": wardrobe_signature,
        "destination": destination.lower(),
        "start": start_date,
        "end": end_date,
        "activities": sorted(normalized_activities),
        "weather": weather,
    }, sort_keys=True)
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()

    if cache_key in vacation_cache:
        cached = vacation_cache[cache_key].copy()
        cached["from_cache"] = True
        return cached

    items_list = []
    for i, item in enumerate(wardrobe):
        items_list.append(
            f"[{i}] {item.get('name', 'Item')} — {item.get('category', '?')}, "
            f"Color: {item.get('color', '?')}, Style: {item.get('style', '?')}, "
            f"Sub: {item.get('subcategory', '?')}, Season: {item.get('season', '?')}"
        )
    items_text = "\n".join(items_list)

    profile_text = ""
    if style_profile:
        parts = []
        if style_profile.get("vibe"):
            parts.append(f"Style vibe: {style_profile['vibe']}")
        if style_profile.get("colors"):
            parts.append(f"Preferred colors: {', '.join(style_profile['colors'])}")
        if style_profile.get("avoid"):
            avoid_colors = [c for c in style_profile['avoid'] if c != 'none']
            if avoid_colors:
                parts.append(f"Colors to AVOID: {', '.join(avoid_colors)}")
        if parts:
            profile_text = "\n\nUSER STYLE PROFILE:\n" + "\n".join(f"- {p}" for p in parts)

    activities_text = ", ".join(normalized_activities)

    # Compute minimum packing counts based on trip length
    min_tops = max(3, (days + 1) // 2)  # ceil(days/2), at least 3
    min_bottoms = max(2, (days + 3) // 4)  # ceil(days/4), at least 2

    # Check if walking-friendly shoes are required
    needs_walking_shoes = any(a in normalized_activities for a in ["sightseeing", "casual", "hiking", "workout"])
    needs_dressy_shoes = any(a in normalized_activities for a in ["dinner", "nightlife", "business"])

    if needs_walking_shoes and needs_dressy_shoes:
        shoe_guidance = "Pack BOTH comfortable walking shoes (sneakers, flats, loafers, boots) AND one dressier pair (heels, dress shoes, or smart boots) - the trip has both walking-heavy and dressy activities."
    elif needs_walking_shoes:
        shoe_guidance = "Pack at least 1 pair of COMFORTABLE WALKING shoes (sneakers, flats, loafers, or comfortable boots). Do NOT pack only heels or dress shoes - the activities require walking comfort."
    elif needs_dressy_shoes:
        shoe_guidance = "Pack dressier shoes appropriate for dinner/nightlife/business (heels, dress boots, loafers)."
    else:
        shoe_guidance = "Pack at least 1 versatile pair of shoes that suits the activities."

    # Outerwear guidance by weather
    if weather in ["hot", "warm"]:
        outerwear_guidance = "0-1 light outerwear pieces (light cardigan or linen blazer only if useful for evening)"
    elif weather == "moderate":
        outerwear_guidance = "1 outerwear piece (light jacket or blazer)"
    elif weather == "cool":
        outerwear_guidance = "1-2 outerwear pieces (jacket + warmer layer)"
    elif weather == "cold":
        outerwear_guidance = "2 outerwear pieces (warm coat + extra layer like a sweater or vest)"
    else:
        outerwear_guidance = "0-1 outerwear pieces depending on need"

    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "Claude API not configured on the backend."},
        )

    try:
        prompt = f"""You are an expert travel stylist for "Styligma ✧".

The user is planning a trip and wants you to pack their suitcase using items from THEIR EXISTING WARDROBE.

TRIP DETAILS:
- Destination: {destination}
- Dates: {start_date} → {end_date} ({days} days)
- Weather: {weather}
- Planned activities: {activities_text}
{profile_text}

THEIR WARDROBE:
{items_text}

TASK:
1. Build a PACKING LIST from items in their wardrobe following the MINIMUM COUNTS below. Pick versatile pieces that mix and match - don't just pack everything.
2. Plan an OUTFIT FOR EACH DAY of the trip using only items from the packing list. Outfits should match the activity planned for that day. Vary outfits - don't repeat the same combo across days.
3. Identify any MISSING ITEMS the user should consider bringing - only items NOT in their wardrobe that they'd genuinely need.

PACKING MINIMUMS (NON-NEGOTIABLE for a {days}-day trip):
- Tops: AT LEAST {min_tops} (more if dressier activities require variety)
- Bottoms: AT LEAST {min_bottoms} (jeans, trousers, shorts, skirts - bottoms can be re-worn across days)
- Shoes: {shoe_guidance}
- Outerwear: {outerwear_guidance}
- Accessories: 1-4 versatile pieces (bag, jewelry, sunglasses, hat, scarf)

HARD RULES - NEVER VIOLATE:
- NEVER pack fewer than {min_tops} tops or {min_bottoms} bottoms for this trip
- For sightseeing or casual activities: ALWAYS include comfortable walking shoes - heels-only is WRONG for sightseeing
- For beach activities: include items appropriate for warm weather and water (or flag swimwear as missing)
- For business/dinner: include a polished/elevated piece (blazer, dress, or similar)
- Each daily outfit needs at minimum a top and a bottom (or a full dress)

STYLING RULES:
- Color harmony across packed items so they mix and match (stick to 1 cohesive palette + 1-2 neutrals)
- Re-wear bottoms and outerwear across days while VARYING TOPS - that's how real packing works
- Match outfits to the planned activity for that day
- Beach day = lightweight/swim-friendly; Dinner = elevated; Business = formal; Hiking = sporty; Sightseeing = comfortable shoes + breathable layers

Return ONLY this JSON, nothing else:
{{
  "trip": {{
    "destination": "{destination}",
    "days": {days},
    "weather_summary": "Short weather description"
  }},
  "packing_list": {{
    "tops": [item index numbers],
    "bottoms": [item index numbers],
    "shoes": [item index numbers],
    "outerwear": [item index numbers],
    "accessories": [item index numbers]
  }},
  "daily_outfits": [
    {{
      "day": 1,
      "activity": "main activity for this day",
      "item_indices": [list of indices from the packing list],
      "note": "Short styling note"
    }}
  ],
  "missing_items": ["light rain jacket", "swimwear if beach", "..."],
  "explanation": "Why this packing list works for this trip - 1-2 sentences"
}}

IMPORTANT:
- item indices must be valid numbers from the wardrobe list above
- daily_outfits must have exactly {days} entries
- only suggest missing_items the user genuinely needs and doesn't already have
- ONLY JSON, no other text"""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text.strip()
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            raise ValueError("Claude didn't return valid JSON")

        result = json.loads(match.group(0))

        def clean_indices(arr):
            return [i for i in (arr or []) if isinstance(i, int) and 0 <= i < len(wardrobe)]

        packing = result.get("packing_list", {})
        cleaned_packing = {
            "tops": clean_indices(packing.get("tops")),
            "bottoms": clean_indices(packing.get("bottoms")),
            "shoes": clean_indices(packing.get("shoes")),
            "outerwear": clean_indices(packing.get("outerwear")),
            "accessories": clean_indices(packing.get("accessories")),
        }

        all_packed = set()
        for cat_list in cleaned_packing.values():
            all_packed.update(cat_list)

        cleaned_outfits = []
        for outfit in result.get("daily_outfits", []):
            valid_idx = [i for i in clean_indices(outfit.get("item_indices")) if i in all_packed]
            if valid_idx:
                cleaned_outfits.append({
                    "day": outfit.get("day"),
                    "activity": outfit.get("activity", ""),
                    "item_indices": valid_idx,
                    "note": outfit.get("note", ""),
                })

        final_result = {
            "trip": result.get("trip", {"destination": destination, "days": days, "weather_summary": ""}),
            "packing_list": cleaned_packing,
            "daily_outfits": cleaned_outfits,
            "missing_items": result.get("missing_items", []),
            "explanation": result.get("explanation", ""),
            "from_cache": False,
        }

        if len(vacation_cache) > 200:
            vacation_cache.pop(next(iter(vacation_cache)))
        vacation_cache[cache_key] = final_result

        return final_result

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500,
            content={"error": "Couldn't parse AI response. Try again."},
        )
    except Exception as e:
        print(f"Vacation list generation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return Path("privacy.html").read_text(encoding="utf-8")


@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return Path("terms.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
