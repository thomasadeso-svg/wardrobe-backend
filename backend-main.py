from fastapi import FastAPI, UploadFile, File
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
from typing import Dict, Any
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
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# -----------------------------
# SPEED SETTINGS
# -----------------------------
REMOVE_BG_MAX = int(os.getenv("REMOVE_BG_MAX", "1024")) # rembg input size
ANALYZE_MAX = int(os.getenv("ANALYZE_MAX", "768")) # Claude input size
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "82")) # speed/quality balance

# -----------------------------
# IMPORTANT: Lazy-load rembg session to avoid Railway boot crash
# -----------------------------
fast_session = None

def get_fast_session():
global fast_session
if fast_session is None:
# lightweight faster model
fast_session = new_session("u2netp")
return fast_session

# -----------------------------
# Small in-memory caches (optional but fast for repeated tests)
# -----------------------------
REMOVE_BG_CACHE: Dict[str, str] = {}
ANALYZE_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LIMIT = 80 # keep small for Railway memory

ALLOWED_CATEGORIES = {"top", "bottom", "shoes", "outerwear", "bag", "jewelry", "accessory"}

def _hash_bytes(b: bytes) -> str:
return hashlib.sha256(b).hexdigest()

def _cache_set(cache: dict, key: str, value):
cache[key] = value
if len(cache) > CACHE_LIMIT:
first = next(iter(cache))
cache.pop(first, None)

def _extract_json(text: str) -> Dict[str, Any]:
if not text:
return {}
m = re.search(r"\{.*\}", text, re.DOTALL)
if not m:
return {}
try:
return json.loads(m.group(0))
except:
return {}

def _resize_to_max(contents: bytes, max_side: int) -> bytes:
"""
Resize + convert to JPEG for faster upload/inference.
"""
img = Image.open(io.BytesIO(contents))
img = img.convert("RGB")
if max(img.size) > max_side:
img.thumbnail((max_side, max_side), Image.LANCZOS)
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
return buf.getvalue()


@app.get("/")
async def root():
return {
"status": "live",
"service": "styligma-v2",
"endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit", "/match-item", "/privacy", "/terms"],
"rembg": True,
"claude": bool(ANTHROPIC_API_KEY),
}


# =========================================================
# REMOVE BACKGROUND (FAST + STABLE)
# =========================================================
@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
try:
input_data = await file.read()
if not input_data:
return JSONResponse(status_code=400, content={"success": False, "error": "Empty file"})

# cache hit
key = _hash_bytes(input_data)
if key in REMOVE_BG_CACHE:
return {
"success": True,
"image": REMOVE_BG_CACHE[key],
"method": "rembg-local-fast-cache",
}

# Resize to max 1024 BEFORE rembg (cuts big time)
resized_jpg = _resize_to_max(input_data, REMOVE_BG_MAX)

# Lazy session init (prevents deployment crash)
session = get_fast_session()

# remove bg
output_data = remove(resized_jpg, session=session)

# Boost color saturation (kept from your code, but slightly lighter)
img = Image.open(io.BytesIO(output_data)).convert("RGBA")
r, g, b, a = img.split()
rgb_img = Image.merge("RGB", (r, g, b))

enhancer = ImageEnhance.Color(rgb_img)
rgb_img = enhancer.enhance(1.18)
enhancer = ImageEnhance.Contrast(rgb_img)
rgb_img = enhancer.enhance(1.08)

r2, g2, b2 = rgb_img.split()
img = Image.merge("RGBA", (r2, g2, b2, a))

buf = io.BytesIO()
img.save(buf, format="PNG", optimize=True)
base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

data_url = f"data:image/png;base64,{base64_image}"
_cache_set(REMOVE_BG_CACHE, key, data_url)

return {
"success": True,
"image": data_url,
"method": "rembg-local-fast",
}

except Exception as e:
return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# =========================================================
# ANALYZE CLOTHING (FAST + "CUP BUG" FIXED)
# =========================================================
@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
try:
if not client:
return {
"rejected": True,
"reason": "AI not available. Please try again later with a clear clothing photo."
}

contents = await file.read()
if not contents:
return JSONResponse(status_code=400, content={"rejected": True, "reason": "Empty file"})

# cache hit
key = _hash_bytes(contents)
if key in ANALYZE_CACHE:
return JSONResponse(content=ANALYZE_CACHE[key])

# SPEED: resize for Claude
resized_jpg = _resize_to_max(contents, ANALYZE_MAX)
base64_image = base64.b64encode(resized_jpg).decode("utf-8")

message = client.messages.create(
model="claude-sonnet-4-20250514",
max_tokens=420, # faster
messages=[{
"role": "user",
"content": [
{
"type": "image",
"source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image},
},
{
"type": "text",
"text": """You are a strict fashion wardrobe gatekeeper. Your ONLY job is to accept real clothing items and fashion accessories, and REJECT everything else.

ABSOLUTE REJECT (must reject):
cups/mugs, food, drinks, electronics, laptops, phones, furniture, rooms, landscapes, plants, toys, tools, animals, people/selfies/faces/body parts, or ANY object that is not worn on the body.
If blurry or unclear: REJECT.

When in doubt: REJECT.

If REJECTED, return ONLY this JSON:
{"rejected": true, "reason": "This doesn't look like a clothing item or accessory. Please photograph a single piece of clothing, shoes, bag, jewelry, or accessory."}

If ACCEPTED, return ONLY this JSON:
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

response_text = message.content[0].text.strip() if message.content else ""
result = _extract_json(response_text)

# HARD SERVER VALIDATION (prevents cup => clothing permanently)
if not isinstance(result, dict) or "rejected" not in result:
safe = {
"rejected": True,
"reason": "Could not analyze this image. Please try again with a clear photo of a clothing item."
}
_cache_set(ANALYZE_CACHE, key, safe)
return JSONResponse(content=safe)

if result.get("rejected") is True:
safe = {"rejected": True, "reason": result.get("reason") or "Not a clothing item or accessory."}
_cache_set(ANALYZE_CACHE, key, safe)
return JSONResponse(content=safe)

cat = str(result.get("category", "")).lower().strip()
if cat not in ALLOWED_CATEGORIES:
safe = {
"rejected": True,
"reason": "This doesn't look like a clothing item or accessory. Please photograph a single fashion item."
}
_cache_set(ANALYZE_CACHE, key, safe)
return JSONResponse(content=safe)

# normalize output so frontend never crashes
result["category"] = cat
if not isinstance(result.get("colors"), list):
result["colors"] = [result.get("color")] if result.get("color") else []
if not isinstance(result.get("season"), list):
result["season"] = ["all-season"]

_cache_set(ANALYZE_CACHE, key, result)
return JSONResponse(content=result)

except json.JSONDecodeError:
safe = {
"rejected": True,
"reason": "Could not analyze this image. Please try again with a clear photo of a clothing item."
}
return JSONResponse(content=safe)
except Exception as e:
return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# GENERATE OUTFIT (UNCHANGED FROM YOUR CODE)
# =========================================================
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
model="claude-sonnet-4-20250514",
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


# =========================================================
# MATCH ITEM (UNCHANGED)
# =========================================================
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


@app.get("/privacy", response_class=HTMLResponse)
async def privacy():
return Path("privacy.html").read_text(encoding="utf-8")


@app.get("/terms", response_class=HTMLResponse)
async def terms():
return Path("terms.html").read_text(encoding="utf-8")


if __name__ == "__main__":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

