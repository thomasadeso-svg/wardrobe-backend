from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
import base64
from typing import List, Optional
import json
import random
import requests
from io import BytesIO

app = FastAPI(title="AI Wardrobe API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
REMOVEBG_API_KEY = os.environ.get("REMOVEBG_API_KEY")  # NEW
AI_AVAILABLE = False
BG_REMOVAL_AVAILABLE = False

if API_KEY and API_KEY.startswith("sk-ant-"):
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        AI_AVAILABLE = True
        print("✅ Claude AI enabled")
    except Exception as e:
        print(f"⚠️  Claude AI initialization failed: {e}")

if REMOVEBG_API_KEY:
    BG_REMOVAL_AVAILABLE = True
    print("✅ Remove.bg enabled")
else:
    print("⚠️  REMOVEBG_API_KEY not set - background removal disabled")

class OutfitRequest(BaseModel):
    wardrobe: List[dict]
    occasion: str
    weather: Optional[str] = "moderate"
    timestamp: Optional[int] = None
    seed: Optional[float] = None
    avoid_previous: Optional[List[str]] = []

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Wardrobe API",
        "ai_enabled": AI_AVAILABLE,
        "bg_removal_enabled": BG_REMOVAL_AVAILABLE,
        "endpoints": {
            "POST /analyze-clothing": "Analyze clothing images",
            "POST /generate-outfit": "Generate outfit suggestions",
            "POST /remove-background": "Remove image background"
        }
    }

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from clothing image using Remove.bg API"""
    
    if not BG_REMOVAL_AVAILABLE:
        # Return original image if Remove.bg not configured
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        return {
            "image_base64": base64_image,
            "note": "Background removal not configured - returning original image"
        }
    
    try:
        contents = await file.read()
        
        # Call Remove.bg API
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': BytesIO(contents)},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVEBG_API_KEY},
            timeout=30
        )
        
        if response.status_code == 200:
            # Convert processed image to base64
            processed_image = base64.b64encode(response.content).decode('utf-8')
            return {
                "image_base64": processed_image,
                "success": True
            }
        else:
            # Return original if Remove.bg fails
            print(f"Remove.bg error: {response.status_code} - {response.text}")
            base64_image = base64.b64encode(contents).decode('utf-8')
            return {
                "image_base64": base64_image,
                "note": f"Background removal failed: {response.status_code}",
                "success": False
            }
            
    except Exception as e:
        print(f"Background removal error: {e}")
        # Return original image on error
        base64_image = base64.b64encode(contents).decode('utf-8')
        return {
            "image_base64": base64_image,
            "note": f"Background removal failed: {str(e)}",
            "success": False
        }

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze a clothing image"""
    
    if not AI_AVAILABLE:
        return fallback_analyze()
    
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        media_type = file.content_type or "image/jpeg"
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this clothing item and return ONLY a JSON object (no markdown) with:
{
  "category": "one of: top, bottom, dress, outerwear, shoes, accessory",
  "color": "primary color name",
  "style": "style description (casual, formal, sporty, etc.)",
  "season": "one of: spring, summer, fall, winter, all-season",
  "tags": ["tag1", "tag2", "tag3"]
}"""
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        return result
        
    except Exception as e:
        print(f"Error analyzing clothing: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-outfit")
async def generate_outfit(request: OutfitRequest):
    """Generate outfit with variety"""
    
    if not AI_AVAILABLE:
        return fallback_outfit_random(request.wardrobe, request.avoid_previous)
    
    try:
        wardrobe_description = "\n".join([
            f"Item {idx}: {item.get('category', 'item')} - {item.get('color', 'unknown')} {item.get('style', '')} ({item.get('season', 'any')} season)"
            for idx, item in enumerate(request.wardrobe)
        ])
        
        variety_instruction = ""
        if request.avoid_previous:
            variety_instruction = f"\n\nDo NOT suggest: {', '.join(request.avoid_previous)}. Create DIFFERENT outfit."
        
        random_seed = random.randint(1, 1000)
        variety_instruction += f"\n\nVariety seed: {random_seed}"
        
        # ADD PATTERN MATCHING RULE
        pattern_rule = "\n\nIMPORTANT: Avoid combining busy patterns (stripes + leopard, polka dots + plaid, etc.). If using a patterned piece, pair with solid colors."
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Given wardrobe:
{wardrobe_description}

Create outfit for: {request.occasion}
Weather: {request.weather}
{variety_instruction}
{pattern_rule}

RULES:
1. Select by index (0 to {len(request.wardrobe)-1})
2. Create UNIQUE combination
3. Good color coordination
4. Weather appropriate
5. NO pattern clashing

Return JSON:
{{
  "outfit": [{{"item_index": 0, "reason": "why"}}],
  "explanation": "color theory + style reasoning",
  "styling_tips": ["tip1", "tip2"]
}}"""
                }
            ],
        )
        
        response_text = message.content[0].text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return fallback_outfit_random(request.wardrobe, request.avoid_previous)

def fallback_analyze():
    return {
        "category": "top",
        "color": "unknown",
        "style": "casual",
        "season": "all-season",
        "tags": ["manual-review"]
    }

def fallback_outfit_random(wardrobe, avoid_previous=[]):
    if len(wardrobe) < 2:
        return {
            "outfit": [{"item_index": 0, "reason": "Only item"}],
            "explanation": "Need more items",
            "styling_tips": ["Add more items"]
        }
    
    available = list(range(len(wardrobe)))
    random.shuffle(available)
    num_items = min(random.randint(2, 4), len(wardrobe))
    selected = available[:num_items]
    
    return {
        "outfit": [{"item_index": idx, "reason": f"Item {idx+1}"} for idx in selected],
        "explanation": "Random outfit for variety",
        "styling_tips": ["Try regenerating"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 AI Wardrobe Backend")
    print(f"AI: {'✅' if AI_AVAILABLE else '❌'}")
    print(f"BG Removal: {'✅' if BG_REMOVAL_AVAILABLE else '❌'}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)


