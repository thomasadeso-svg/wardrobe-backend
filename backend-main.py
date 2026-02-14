from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        "endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit"],
        "rembg": True,
        "claude": bool(ANTHROPIC_API_KEY),
    }


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        input_data = await file.read()
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
            "method": "rembg-local-free",
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
                        "text": """You are a fashion wardrobe assistant. Analyze this image.

FIRST: Determine if this is a clothing item, shoes, bag, jewelry, or fashion accessory.

If this is NOT a fashion/clothing item (e.g. food, animals, furniture, people without focus on clothes, random objects, electronics, etc.), return ONLY this JSON:
{"rejected": true, "reason": "This doesn't look like a clothing item or accessory. Please photograph a piece of clothing, shoes, bag, jewelry, or accessory."}

If this IS a valid fashion item, return ONLY this JSON:
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

    if len(wardrobe) < 2:
        return {
            "outfit": [],
            "explanation": "Add more items! You need at least one top and one bottom.",
            "styling_tip": "",
        }

    # Build numbered item list for Claude
    items_list = []
    for i, item in enumerate(wardrobe):
        items_list.append(
            f"[{i}] {item.get('name', 'Item')} — {item.get('category', '?')}, "
            f"Color: {item.get('color', '?')}, Style: {item.get('style', '?')}, "
            f"Sub: {item.get('subcategory', '?')}"
        )
    items_text = "\n".join(items_list)

    # Format previous outfits so Claude avoids them
    avoid_text = ""
    if previous_outfits:
        avoid_lines = []
        for prev in previous_outfits[-5:]:  # Only last 5 to save tokens
            avoid_lines.append(str(prev))
        avoid_text = f"\n\nAVOID these combinations (already shown):\n" + "\n".join(avoid_lines)

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

Pick the BEST outfit for:
- Occasion: {occasion}
- Weather: {weather}

RULES:
- MUST pick 1 top + 1 bottom (required)
- Pick shoes if available
- Pick 1 bag/jewelry/accessory if it complements the look
- If cold weather, add outerwear
- DO NOT repeat previous combinations — pick DIFFERENT items
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
            if pick_text.startswith("```"):
                pick_text = pick_text.split("\n", 1)[1]
                pick_text = pick_text.rsplit("```", 1)[0].strip()

            pick_data = json.loads(pick_text)
            indices = pick_data.get("selected_indices", [])
            valid_indices = [i for i in indices if 0 <= i < len(wardrobe)]

            if valid_indices:
                return {
                    "outfit": [{"item_index": i} for i in valid_indices],
                    "explanation": pick_data.get("explanation", "A curated look styled by AI."),
                    "styling_tip": pick_data.get("styling_tip", "Own it with confidence."),
                }
        except Exception as e:
            print(f"AI outfit selection failed: {e}")

    # Fallback random
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
    if outerwear and weather == "cold":
        selected.append({"item_index": outerwear[0]})

    return {
        "outfit": selected,
        "explanation": "A fresh combination from your wardrobe.",
        "styling_tip": "Mix textures for a balanced silhouette.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
