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

# Claude AI - our only paid API (and it's cheap)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


@app.get("/")
async def root():
    return {
        "status": "live",
        "service": "wardrobe-ai",
        "endpoints": ["/remove-background", "/analyze-clothing", "/generate-outfit"],
        "rembg": True,
        "claude": bool(ANTHROPIC_API_KEY),
    }


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """FREE background removal using rembg on server CPU"""
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
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Claude AI categorizes the clothing item"""
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": file.content_type or "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this clothing item. Return ONLY valid JSON with these fields:
{
  "category": "top" or "bottom" or "shoes" or "outerwear" or "accessory",
  "subcategory": "e.g. t-shirt, jeans, sneakers, jacket, hat",
  "color": "primary color name",
  "colors": ["primary", "secondary if any"],
  "style": "casual" or "formal" or "sporty" or "streetwear" or "elegant",
  "season": ["spring", "summer", "fall", "winter"],
  "fabric_guess": "e.g. cotton, denim, leather, polyester",
  "name": "Short descriptive name like 'Black Slim Jeans' or 'White Cotton Tee'"
}
Return ONLY the JSON, no other text.""",
                        },
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()
        # Clean up response if it has markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0].strip()

        return JSONResponse(content=json.loads(response_text))
    except json.JSONDecodeError:
        return {"category": "top", "color": "unknown", "style": "casual", "name": "Clothing Item"}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": str(e)}
        )


@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    """Smart outfit generation with Claude AI styling advice"""
    wardrobe = request.get("wardrobe", [])
    occasion = request.get("occasion", "casual")
    weather = request.get("weather", "moderate")

    # Categorize items
    tops = [i for i, item in enumerate(wardrobe) if item.get("category") in ["top"]]
    bottoms = [i for i, item in enumerate(wardrobe) if item.get("category") in ["bottom"]]
    shoes = [i for i, item in enumerate(wardrobe) if item.get("category") in ["shoes"]]
    outerwear = [i for i, item in enumerate(wardrobe) if item.get("category") in ["outerwear"]]

    if not tops or not bottoms:
        return {
            "outfit": [],
            "explanation": "Add more items to your wardrobe! You need at least one top and one bottom.",
            "styling_tip": "",
        }

    # Shuffle for variety
    random.shuffle(tops)
    random.shuffle(bottoms)
    random.shuffle(shoes)
    random.shuffle(outerwear)

    # Build outfit
    selected = [{"item_index": tops[0]}, {"item_index": bottoms[0]}]
    if shoes:
        selected.append({"item_index": shoes[0]})
    if outerwear and weather == "cold":
        selected.append({"item_index": outerwear[0]})

    # Use Claude for smart styling tip if available
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
                max_tokens=150,
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a fashion stylist. For this outfit: {items_desc}. Occasion: {occasion}. Weather: {weather}. Give a one-sentence styling tip and a one-sentence explanation of why this works. Return JSON: {{\"tip\": \"...\", \"explanation\": \"...\"}}",
                    }
                ],
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

