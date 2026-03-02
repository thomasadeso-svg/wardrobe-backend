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
from PIL import Image, ImageEnhance
from rembg import remove

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


@app.get("/")
async def root():
    return {
        "status": "live",
        "service": "styligma",
        "endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit", "/match-item", "/privacy", "/terms"],
        "rembg": True,
        "claude": bool(ANTHROPIC_API_KEY),
    }


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        input_data = await file.read()

        # Resize large images before processing (big speed boost)
        img_input = Image.open(io.BytesIO(input_data))
        max_size = 1024
        if max(img_input.size) > max_size:
            img_input.thumbnail((max_size, max_size), Image.LANCZOS)
            buf_resized = io.BytesIO()
            img_input.save(buf_resized, format="PNG")
            input_data = buf_resized.getvalue()

        # Use default model (best quality) — speed comes from resize above
        output_data = remove(input_data)

        # Boost color saturation for vivid images
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
            "method": "rembg-optimized",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
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
                        "text": """You are a strict fashion wardrobe gatekeeper. Your job is to ONLY accept real clothing items and fashion accessories.

STEP 1 — STRICTLY determine if this image shows ONE of these:
ACCEPTED: shirts, t-shirts, blouses, sweaters, hoodies, jackets, coats, blazers, vests, pants, jeans, shorts, skirts, dresses, shoes, sneakers, boots, sandals, heels, bags, handbags, backpacks, belts, watches, necklaces, bracelets, rings, earrings, sunglasses, scarves, hats, caps, ties, gloves, socks

REJECTED (return rejected=true for ALL of these):
- People, selfies, faces, body parts
- Food, drinks, plates, kitchen items
- Animals, pets
- Cars, vehicles, bikes
- Furniture, rooms, buildings, landscapes
- Electronics, phones, laptops, TVs
- Books, papers, documents
- Plants, flowers, trees
- Toys, games
- Tools, hardware
- Money, cards
- ANY object that is not worn on the body as clothing or fashion accessory
- Blurry, unclear, or unrecognizable images
- Multiple items in one photo (user should photograph ONE item at a time)

If REJECTED, return ONLY:
{"rejected": true, "reason": "This doesn't look like a clothing item or accessory. Please photograph a single piece of clothing, shoes, bag, jewelry, or accessory."}

Be VERY strict. When in doubt, REJECT. Only accept items that are clearly wearable fashion items.

STEP 2 — If this IS a valid single fashion item, return ONLY this JSON:
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
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0].strip()

        result = json.loads(response_text)
        return JSONResponse(content=result)
    except json.JSONDecodeError:
        return {"rejected": False, "category": "top", "color": "unknown", "style": "casual", "name": "Clothing Item"}
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

    # Categorize all wardrobe items
    by_category = {"top": [], "bottom": [], "shoes": [], "outerwear": [], "bag": [], "jewelry": [], "accessory": []}
    for i, item in enumerate(wardrobe):
        cat = item.get("category", "top")
        if cat in by_category:
            by_category[cat].append(i)

    # Build numbered item list for Claude
    items_list = []
    for i, item in enumerate(wardrobe):
        items_list.append(
            f"[{i}] {item.get('name', 'Item')} — category: {item.get('category', '?')}, "
            f"Color: {item.get('color', '?')}, Style: {item.get('style', '?')}, "
            f"Sub: {item.get('subcategory', '?')}"
        )
    items_text = "\n".join(items_list)

    # Show available categories to Claude
    available_cats = {cat: idxs for cat, idxs in by_category.items() if idxs}
    category_summary = "\n".join(f"  {cat}: indices {idxs}" for cat, idxs in available_cats.items())

    # Format previous outfits so Claude avoids them
    avoid_text = ""
    if previous_outfits:
        avoid_lines = []
        for prev in previous_outfits[-5:]:
            avoid_lines.append(str(prev))
        avoid_text = f"\n\nAVOID these exact combinations (already shown):\n" + "\n".join(avoid_lines)

    # Style profile personalization
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
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""You are an expert fashion stylist for "Styligma ✧".

WARDROBE:
{items_text}

ITEMS GROUPED BY CATEGORY:
{category_summary}
{avoid_text}
{profile_text}

Pick the BEST outfit for:
- Occasion: {occasion}
- Weather: {weather}

STRICT SELECTION RULES — YOU MUST FOLLOW THESE EXACTLY:
1. Pick EXACTLY 1 "top" item (REQUIRED)
2. Pick EXACTLY 1 "bottom" item (REQUIRED)
3. Pick EXACTLY 1 "shoes" item (if available)
4. Weather rules for outerwear:
   - "cold" weather: Pick EXACTLY 1 "outerwear" item (REQUIRED)
   - "moderate" weather: Optionally pick 1 "outerwear" item
   - "hot" weather: Do NOT pick any outerwear
5. Optionally pick 1 accessory/jewelry/bag item (max 1)

ABSOLUTE RULES — VIOLATION IS FORBIDDEN:
- NEVER select 2 items from the same category
- NEVER select 2 tops, 2 bottoms, 2 shoes, or 2 outerwear pieces
- Each category appears AT MOST ONCE
- The outfit should have 3-5 items total, never more

VARIETY: Pick DIFFERENT items than previous outfits shown above.

STYLE: Focus on color harmony, occasion-appropriate pairings, and texture contrast.

Return ONLY this JSON:
{{
  "selected_indices": [one index per category, 3-5 total],
  "explanation": "Why these pieces work — mention colors and textures",
  "styling_tip": "One actionable tip for wearing this outfit"
}}

Double-check: each index must come from a DIFFERENT category. ONLY JSON, nothing else.""",
                }],
            )

            pick_text = pick_response.content[0].text.strip()
            if pick_text.startswith("```"):
                pick_text = pick_text.split("\n", 1)[1]
                pick_text = pick_text.rsplit("```", 1)[0].strip()

            pick_data = json.loads(pick_text)
            indices = pick_data.get("selected_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(wardrobe)]

            # ENFORCE server-side: no duplicate categories ever
            seen_categories = set()
            deduplicated = []
            for i in valid_indices:
                cat = wardrobe[i].get("category", "unknown")
                if cat not in seen_categories:
                    seen_categories.add(cat)
                    deduplicated.append(i)
            valid_indices = deduplicated

            if valid_indices and len(valid_indices) >= 2:
                return {
                    "outfit": [{"item_index": i} for i in valid_indices],
                    "explanation": pick_data.get("explanation", "A curated look styled by AI."),
                    "styling_tip": pick_data.get("styling_tip", "Own it with confidence."),
                }
        except Exception as e:
            print(f"AI outfit selection failed: {e}")

    # Fallback random — also enforces one per category
    tops = [i for i, item in enumerate(wardrobe) if item.get("category") == "top"]
    bottoms = [i for i, item in enumerate(wardrobe) if item.get("category") == "bottom"]
    shoes = [i for i, item in enumerate(wardrobe) if item.get("category") == "shoes"]
    outerwear = [i for i, item in enumerate(wardrobe) if item.get("category") == "outerwear"]

    if not tops or not bottoms:
        return {"outfit": [], "explanation": "Need at least one top and one bottom.", "styling_tip": ""}

    for lst in [tops, bottoms, shoes, outerwear]:
        random.shuffle(lst)

    selected = [{"item_index": tops[0]}, {"item_index": bottoms[0]}]
    if shoes:
        selected.append({"item_index": shoes[0]})
    if outerwear and weather in ["cold", "moderate"]:
        selected.append({"item_index": outerwear[0]})

    return {
        "outfit": selected,
        "explanation": "A fresh combination from your wardrobe.",
        "styling_tip": "Mix textures for a balanced silhouette.",
    }


@app.post("/match-item")
async def match_item(request: dict):
    """Try Before You Buy: Given a new item's details and the user's wardrobe,
    find which wardrobe items pair well with it and suggest complete outfits."""
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

    # Build wardrobe list
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
                model="claude-sonnet-4-20250514",
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
            if pick_text.startswith("```"):
                pick_text = pick_text.split("\n", 1)[1]
                pick_text = pick_text.rsplit("```", 1)[0].strip()

            result = json.loads(pick_text)

            # Validate indices
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

    # Fallback: simple category matching
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


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
    return Path("privacy.html").read_text(encoding="utf-8")


@app.get("/terms", response_class=HTMLResponse)
async def terms():
    return Path("terms.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

