from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
import io
import base64
import json
import random
from PIL import Image
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
        "service": "stylenigma",
        "endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit"],
        "rembg": True,
        "claude": bool(ANTHROPIC_API_KEY),
    }


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        input_data = await file.read()
        output_data = remove(input_data)
        base64_image = base64.b64encode(output_data).decode("utf-8")
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

    tops = [i for i, item in enumerate(wardrobe) if item.get("category") == "top"]
    bottoms = [i for i, item in enumerate(wardrobe) if item.get("category") == "bottom"]
    shoes = [i for i, item in enumerate(wardrobe) if item.get("category") == "shoes"]
    outerwear = [i for i, item in enumerate(wardrobe) if item.get("category") == "outerwear"]
    bags = [i for i, item in enumerate(wardrobe) if item.get("category") == "bag"]
    jewelry = [i for i, item in enumerate(wardrobe) if item.get("category") == "jewelry"]
    accessories = [i for i, item in enumerate(wardrobe) if item.get("category") == "accessory"]

    if not tops or not bottoms:
        return {
            "outfit": [],
            "explanation": "Add more items to your wardrobe! You need at least one top and one bottom.",
            "styling_tip": "",
        }

    for lst in [tops, bottoms, shoes, outerwear, bags, jewelry, accessories]:
        random.shuffle(lst)

    selected = [{"item_index": tops[0]}, {"item_index": bottoms[0]}]
    if shoes:
        selected.append({"item_index": shoes[0]})
    if outerwear and weather == "cold":
        selected.append({"item_index": outerwear[0]})
    if bags:
        selected.append({"item_index": bags[0]})
    if jewelry:
        selected.append({"item_index": jewelry[0]})
    if accessories:
        selected.append({"item_index": accessories[0]})

    styling_tip = "Mix textures and proportions for a balanced silhouette."
    explanation = "A curated look from your wardrobe, styled for your day."

    if client and len(wardrobe) >= 2:
        try:
            selected_items = [wardrobe[s["item_index"]] for s in selected]
            items_desc = ", ".join(
                [f"{item.get('name', item.get('color', 'item'))} ({item.get('category', 'piece')})" for item in selected_items]
            )
            tip_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"""You are a fashion stylist for a wardrobe app called "StyleNigma". 
For this outfit: {items_desc}. Occasion: {occasion}. Weather: {weather}. 
Give one styling tip and one explanation of why this works. Be specific about colors and pieces.
Return JSON: {{"tip": "...", "explanation": "..."}}""",
                }],
            )
            tip_text = tip_response.content[0].text.strip()
            if tip_text.startswith("```"):
                tip_text = tip_text.split("\n", 1)[1]
                tip_text = tip_text.rsplit("```", 1)[0].strip()
            tip_data = json.loads(tip_text)
            styling_tip = tip_data.get("tip", styling_tip)
            explanation = tip_data.get("explanation", explanation)
        except:
            pass

    return {
        "outfit": selected,
        "explanation": explanation,
        "styling_tip": styling_tip,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
