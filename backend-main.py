from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
import anthropic
import os
import io
import asyncio
import base64
import json
import random
import re
import hashlib
import requests
from PIL import Image, ImageEnhance
from background_removal import remove_background_bytes
from video_scan import router as video_scan_router

app = FastAPI()

app.include_router(video_scan_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

 # Sessions load lazily on first use, through background_removal.py.

vacation_cache = {}

VALID_ACTIVITIES = {"beach", "dinner", "sightseeing", "nightlife", "hiking", "business", "casual", "workout"}


@app.get("/")
async def root():
    return {
        "status": "live",
        "service": "styligma-v2",
        "outfit_api_version": 2,
        "compatibility_api_version": 1,
        "outfit_features": ["batches", "exact_item_anchor", "hot_outerwear_anchor"],
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

        output_data = await asyncio.to_thread(remove_background_bytes, input_data, "u2netp")

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
  "category": "top" or "bottom" or "dress" or "shoes" or "outerwear" or "bag" or "jewelry" or "accessory",
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
    # The synchronous SDK runs off the event loop. No implicit SDK retries.
    from outfit_engine import build_batch

    def ask_ai(prompt):
        response = client.with_options(max_retries=0, timeout=10.0).messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1400 if request.get("batch_size", 1) != 1 else 500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("Invalid outfit JSON")
        return json.loads(match.group(0))

    try:
        return await asyncio.to_thread(build_batch, request, ask_ai if client else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/match-item")
async def match_item(request: dict):
    from wardrobe_compatibility import assess
    try:
        return await asyncio.to_thread(assess, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/vacation-list")
async def vacation_list(request: dict):
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

    min_tops = max(3, (days + 1) // 2)
    min_bottoms = max(2, (days + 3) // 4)

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
