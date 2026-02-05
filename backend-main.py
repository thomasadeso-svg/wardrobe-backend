from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
import base64
from typing import List, Optional
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Wardrobe API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic client and check if API key is valid
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
AI_AVAILABLE = False

if API_KEY and API_KEY.startswith("sk-ant-"):
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        AI_AVAILABLE = True
        print("✅ Claude AI enabled - API key loaded successfully")
    except Exception as e:
        print(f"⚠️  Claude AI initialization failed: {e}")
        AI_AVAILABLE = False
else:
    print("⚠️  ANTHROPIC_API_KEY not found or invalid - running in fallback mode")
    print(f"    Current key value: {API_KEY[:20] if API_KEY else 'None'}...")

class ClothingItem(BaseModel):
    category: str
    color: str
    style: str
    season: str
    tags: List[str]

class OutfitRequest(BaseModel):
    wardrobe: List[dict]
    occasion: str
    weather: Optional[str] = "moderate"

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Wardrobe API",
        "ai_enabled": AI_AVAILABLE,
        "endpoints": {
            "POST /analyze-clothing": "Analyze clothing images",
            "POST /generate-outfit": "Generate outfit suggestions"
        }
    }

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze a clothing image and extract details"""
    
    if not AI_AVAILABLE:
        return fallback_analyze()
    
    try:
        # Read and encode image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Determine media type
        media_type = file.content_type or "image/jpeg"
        
        # Call Claude API
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
                            "text": """Analyze this clothing item and return ONLY a JSON object (no markdown, no extra text) with:
{
  "category": "one of: top, bottom, dress, outerwear, shoes, accessory",
  "color": "primary color name",
  "style": "style description (casual, formal, sporty, etc.)",
  "season": "one of: spring, summer, fall, winter, all-season",
  "tags": ["tag1", "tag2", "tag3"]
}

Tags should include: formality level, patterns, material type, and notable features."""
                        }
                    ],
                }
            ],
        )
        
        # Parse response
        response_text = message.content[0].text
        # Remove markdown code blocks if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        return result
        
    except Exception as e:
        print(f"Error analyzing clothing: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/generate-outfit")
async def generate_outfit(request: OutfitRequest):
    """Generate outfit recommendations based on wardrobe and occasion"""
    
    if not AI_AVAILABLE:
        return fallback_outfit(request.wardrobe)
    
    try:
        wardrobe_description = "\n".join([
            f"- {item.get('category', 'item')}: {item.get('color', 'unknown')} {item.get('style', '')} ({item.get('season', 'any')} season)"
            for item in request.wardrobe
        ])
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this wardrobe:
{wardrobe_description}

Create an outfit for: {request.occasion}
Weather: {request.weather}

Return ONLY a JSON object (no markdown) with:
{{
  "outfit": [
    {{
      "item_index": 0,
      "reason": "why this item works"
    }}
  ],
  "explanation": "overall outfit explanation",
  "styling_tips": ["tip1", "tip2", "tip3"]
}}

Use item_index to refer to items by their position in the wardrobe list (0-indexed)."""
                }
            ],
        )
        
        response_text = message.content[0].text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(response_text)
        
        return result
        
    except Exception as e:
        print(f"Error generating outfit: {e}")
        raise HTTPException(status_code=500, detail=f"Outfit generation failed: {str(e)}")

def fallback_analyze():
    """Fallback when AI is not available"""
    return {
        "category": "top",
        "color": "unknown",
        "style": "casual",
        "season": "all-season",
        "tags": ["manual-review-needed"],
        "note": "AI analysis unavailable - please categorize manually"
    }

def fallback_outfit(wardrobe):
    """Fallback outfit generation"""
    if len(wardrobe) < 2:
        return {
            "outfit": [{"item_index": 0, "reason": "Only item available"}],
            "explanation": "AI unavailable - showing all available items",
            "styling_tips": ["Add more items for better recommendations"]
        }
    
    return {
        "outfit": [
            {"item_index": i, "reason": "Available item"} 
            for i in range(min(3, len(wardrobe)))
        ],
        "explanation": "AI unavailable - showing first available items",
        "styling_tips": ["Enable AI for smart recommendations"]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting AI Wardrobe Backend Server")
    print(f"AI Status: {'✅ ENABLED' if AI_AVAILABLE else '❌ DISABLED (Fallback Mode)'}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
