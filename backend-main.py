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

# ---- Rembg session handling (NO crash on startup) ----
FAST_SESSION = None
FAST_MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp") # fast + lightweight

@app.on_event("startup")
async def startup_event():
"""
Try to warm up rembg, but NEVER crash the server if it fails.
Railway sometimes has no permission/cache issues on cold start.
"""
global FAST_SESSION
try:
# Use /tmp in containers to avoid permission issues
os.environ.setdefault("U2NET_HOME", "/tmp")
FAST_SESSION = new_session(FAST_MODEL_NAME)
print(f"[startup] rembg session ready: {FAST_MODEL_NAME}")
except Exception as e:
FAST_SESSION = None
print(f"[startup] rembg session failed, will fallback to default remove(): {e}")

@app.get("/")
async def root():
return {
"status": "live",
"service": "styligma-v2",
"endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit", "/match-item", "/privacy", "/terms"],
"rembg_session_ready": bool(FAST_SESSION),
"claude": bool(client),
}

# ---------- Helpers ----------
def _extract_json(text: str) -> dict:
# robust JSON extraction
match = re.search(r"\{.*\}", text, re.DOTALL)
if not match:
raise ValueError("No JSON object found in model response")
return json.loads(match.group(0))

ALLOWED_CATEGORIES = {"top", "bottom", "shoes", "outerwear", "bag", "jewelry", "accessory"}
ALLOWED_STYLES = {"casual", "formal", "sporty", "streetwear", "elegant", "bohemian"}

# Strong “not clothing” keywords (fix: cup mistaken as clothing)
FORBIDDEN_KEYWORDS = {
"cup","mug","glass","bottle","plate","fork","spoon","food","drink","coffee","tea",
"phone","laptop","keyboard","mouse","tv","remote","book","toy","tool","furniture",
"chair","table","sofa","plant","flower","pet","dog","cat","car","vehicle"
}

def _looks_forbidden(result: dict) -> bool:
blob = " ".join([
str(result.get("name","")),
str(result.get("subcategory","")),
str(result.get("category","")),
str(result.get("reason","")),
]).lower()
return any(k in blob for k in FORBIDDEN_KEYWORDS)

def _validate_clothing_json(result: dict) -> dict:
"""
If Claude says accepted but the output is suspicious -> force reject.
This fixes “cup = clothing” and other nonsense.
"""
if result.get("rejected") is True:
return result

# Must have required fields and valid enum-like values
cat = str(result.get("category","")).lower().strip()
style = str(result.get("style","")).lower().strip()

if cat not in ALLOWED_CATEGORIES:
return {
"rejected": True,
"reason": "Das Bild wurde nicht eindeutig als Kleidungsstück/Accessoire erkannt. Bitte fotografiere ein einzelnes Kleidungsstück (z.B. Shirt, Hose, Schuhe, Tasche, Schmuck)."
}

if style and style not in ALLOWED_STYLES:
# normalize unknown style to casual (don’t reject for this)
result["style"] = "casual"

# If keywords indicate non-clothing -> reject hard
if _looks_forbidden(result):
return {
"rejected": True,
"reason": "Das sieht nicht nach Kleidung oder einem tragbaren Accessoire aus. Bitte fotografiere ein einzelnes Kleidungsstück, Schuhe, Tasche oder Schmuck."
}

# Ensure minimal fields
result.setdefault("colors", [result.get("color","unknown")])
result.setdefault("season", ["spring", "summer", "fall", "winter"])
result.setdefault("fabric_guess", "unknown")
result.setdefault("subcategory", "item")
result.setdefault("name", "Kleidungsstück")

return result

# ---------- Endpoints ----------
@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
try:
input_data = await file.read()
if not input_data:
raise HTTPException(status_code=400, detail="Empty file")

# Faster: resize a bit more aggressive (1024 -> 768)
img_input = Image.open(io.BytesIO(input_data))
img_input = img_input.convert("RGBA") # normalize
max_size = int(os.getenv("REMBG_MAX_SIZE", "768"))
if max(img_input.size) > max_size:
img_input.thumbnail((max_size, max_size), Image.LANCZOS)

buf_resized = io.BytesIO()
img_input.save(buf_resized, format="PNG", optimize=True)
resized_data = buf_resized.getvalue()

# Use session if available, otherwise fallback
if FAST_SESSION:
output_data = remove(resized_data, session=FAST_SESSION)
method = "rembg-local-fast-session"
else:
output_data = remove(resized_data)
method = "rembg-local-fallback"

# Optional enhance (can disable via env for speed)
enhance = os.getenv("REMBG_ENHANCE", "1") == "1"
img = Image.open(io.BytesIO(output_data)).convert("RGBA")

if enhance:
r, g, b, a = img.split()
rgb_img = Image.merge("RGB", (r, g, b))
rgb_img = ImageEnhance.Color(rgb_img).enhance(1.25)
rgb_img = ImageEnhance.Contrast(rgb_img).enhance(1.08)
r2, g2, b2 = rgb_img.split()
img = Image.merge("RGBA", (r2, g2, b2, a))

buf = io.BytesIO()
img.save(buf, format="PNG", optimize=True)
base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

return {
"success": True,
"image": f"data:image/png;base64,{base64_image}",
"method": method,
"max_size": max_size,
"enhance": enhance,
}

except HTTPException as e:
raise e
except Exception as e:
return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
try:
if not client:
return JSONResponse(
status_code=503,
content={"rejected": True, "reason": "AI Analyse ist aktuell nicht verfügbar (API-Key fehlt)."},
)

contents = await file.read()
if not contents:
raise HTTPException(status_code=400, detail="Empty file")

base64_image = base64.b64encode(contents).decode("utf-8")

message = client.messages.create(
model="claude-sonnet-4-20250514",
max_tokens=512,
messages=[{
"role": "user",
"content": [
{
"type": "image",
"source": {
"type": "base64",
"media_type": file.content_type or "image/jpeg",
"data": base64_image
},
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
result = _extract_json(response_text)

# HARD server-side validation (fixes “cup=clothing”)
result = _validate_clothing_json(result)

return JSONResponse(content=result)

except HTTPException as e:
raise e
except json.JSONDecodeError:
return {"rejected": True, "reason": "Konnte das Bild nicht analysieren. Bitte versuche es erneut mit einem klaren Foto eines Kleidungsstücks."}
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
return {"outfit": [], "explanation": "Add more items! You need at least one top and one bottom.", "styling_tip": ""}

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
avoid_text = "\n\nAVOID these combinations (already shown):\n" + "\n".join(str(p) for p in previous_outfits[-5:])

profile_text = ""
if style_profile:
parts = []
if style_profile.get("vibe"):
parts.append(f"Style vibe: {style_profile['vibe']}")
if style_profile.get("colors"):
parts.append(f"Preferred colors: {', '.join(style_profile['colors'])}")
if style_profile.get("avoid"):
avoid_colors = [c for c in style_profile["avoid"] if c != "none"]
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
- Total items: 3-5, never more

Return ONLY JSON:
{{
"selected_indices": [0, 3, 5],
"explanation": "Why these pieces work — mention specific colors and textures",
"styling_tip": "One specific actionable tip for wearing this outfit"
}}""",
}],
)

pick_text = pick_response.content[0].text.strip()
pick_data = _extract_json(pick_text)
indices = pick_data.get("selected_indices", [])
valid_indices = [i for i in indices if 0 <= i < len(wardrobe)]

seen_categories = set()
dedup = []
for i in valid_indices:
cat = wardrobe[i].get("category", "unknown").lower().strip()
if cat not in seen_categories:
seen_categories.add(cat)
dedup.append(i)

category_order = {"outerwear": 0, "top": 1, "bottom": 2, "shoes": 3, "bag": 4, "jewelry": 5, "accessory": 6}
dedup.sort(key=lambda x: category_order.get(wardrobe[x].get("category", "unknown").lower().strip(), 99))

if dedup:
return {
"outfit": [{"item_index": i} for i in dedup],
"explanation": pick_data.get("explanation", "A curated look styled by AI."),
"styling_tip": pick_data.get("styling_tip", "Own it with confidence."),
}
except Exception as e:
print(f"AI outfit selection failed: {e}")

# fallback
tops = [i for i, it in enumerate(wardrobe) if it.get("category","").lower()=="top"]
bottoms = [i for i, it in enumerate(wardrobe) if it.get("category","").lower()=="bottom"]
shoes = [i for i, it in enumerate(wardrobe) if it.get("category","").lower()=="shoes"]
outerwear = [i for i, it in enumerate(wardrobe) if it.get("category","").lower()=="outerwear"]

if not tops or not bottoms:
return {"outfit": [], "explanation": "Need at least one top and one bottom.", "styling_tip": ""}

random.shuffle(tops); random.shuffle(bottoms); random.shuffle(shoes); random.shuffle(outerwear)

selected = [{"item_index": tops[0]}, {"item_index": bottoms[0]}]
if shoes: selected.append({"item_index": shoes[0]})
if outerwear and weather in ["cold","moderate"]:
selected.insert(0, {"item_index": outerwear[0]})

return {"outfit": selected, "explanation": "A fresh combination from your wardrobe.", "styling_tip": "Mix textures for a balanced silhouette."}


@app.post("/match-item")
async def match_item(request: dict):
new_item = request.get("new_item", {})
wardrobe = request.get("wardrobe", [])
occasion = request.get("occasion", "casual")

if not new_item or len(wardrobe) < 1:
return {"matches": [], "outfits": [], "verdict": "Add more items to your wardrobe first.", "match_count": 0}

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

NEW ITEM: {new_item_desc}

WARDROBE:
{items_text}

Return ONLY JSON:
{{
"match_count": <number>,
"matching_indices": [indices],
"outfits": [{{"wardrobe_indices":[indices], "description":"..."}}],
"verdict": "SMART BUY: ..." or "SKIP: ..." or "MAYBE: ...",
"color_harmony": "...",
"style_fit": "..."
}}""",
}],
)

pick_text = response.content[0].text.strip()
result = _extract_json(pick_text)

valid_matches = [i for i in result.get("matching_indices", []) if 0 <= i < len(wardrobe)]
valid_outfits = []
for outfit in result.get("outfits", [])[:3]:
valid_idx = [i for i in outfit.get("wardrobe_indices", []) if 0 <= i < len(wardrobe)]
if valid_idx:
valid_outfits.append({"wardrobe_indices": valid_idx, "description": outfit.get("description","")})

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

# fallback
new_cat = new_item.get("category", "")
matches = [i for i, it in enumerate(wardrobe) if it.get("category","") != new_cat]
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
# IMPORTANT: run "backend-main:app" in Procfile on Railway (recommended)
uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
