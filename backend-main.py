from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
import requests
from PIL import Image
import io
import base64

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

@app.get("/")
def read_root():
    return {
        "message": "Wardrobe AI Backend is running! 🎨👔",
        "status": "online",
        "features": {
            "claude_ai": bool(ANTHROPIC_API_KEY),
            "removebg": bool(REMOVEBG_API_KEY),
            "replicate": bool(REPLICATE_API_TOKEN)
        }
    }

@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    """Generate outfit suggestions using Claude AI"""
    try:
        wardrobe = request.get("wardrobe", [])
        occasion = request.get("occasion", "casual")
        weather = request.get("weather", "moderate")
        
        if not wardrobe or len(wardrobe) < 2:
            return JSONResponse(content={
                "outfit": [{"item_index": 0}, {"item_index": 1}],
                "explanation": "Add more items to your wardrobe for better suggestions!"
            })
        
        if not client:
            # Fallback if no API key
            return JSONResponse(content={
                "outfit": [{"item_index": 0}, {"item_index": min(1, len(wardrobe)-1)}],
                "explanation": "Random outfit suggestion (AI offline)"
            })
        
        # Build wardrobe text
        wardrobe_text = "\n".join([
            f"{i}. {item.get('category', 'item')}: {item.get('color', 'colored')} {item.get('style', 'style')}"
            for i, item in enumerate(wardrobe)
        ])
        
        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this wardrobe:
{wardrobe_text}

Create an outfit for {occasion} in {weather} weather.

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "outfit": [
    {{"item_index": 0}},
    {{"item_index": 1}}
  ],
  "explanation": "Brief explanation of why this outfit works"
}}

Use item indices 0 to {len(wardrobe)-1}."""
                }
            ],
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        import json
        result = json.loads(response_text)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Outfit generation error: {str(e)}")
        # Fallback response
        return JSONResponse(content={
            "outfit": [
                {"item_index": 0},
                {"item_index": min(1, len(wardrobe)-1)}
            ],
            "explanation": "Here's a suggested outfit combination!"
        })

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from image using Remove.bg API"""
    try:
        contents = await file.read()
        
        if not REMOVEBG_API_KEY:
            print("❌ No Remove.bg API key found")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Remove.bg API key not configured",
                    "image": None
                }
            )
        
        print(f"📤 Calling Remove.bg API... (file size: {len(contents)} bytes)")
        
        # Call Remove.bg API
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': ('image.jpg', contents, 'image/jpeg')},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVEBG_API_KEY},
            timeout=60  # Increased timeout
        )
        
        print(f"📥 Remove.bg response status: {response.status_code}")
        
        if response.status_code == 200:
            result_base64 = base64.b64encode(response.content).decode('utf-8')
            print("✅ Background removed successfully!")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{result_base64}",
                "method": "removebg"
            })
        elif response.status_code == 403:
            print("❌ Remove.bg API key invalid or quota exceeded")
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error": "Remove.bg API key invalid or quota exceeded. Check your account at remove.bg",
                    "image": None
                }
            )
        elif response.status_code == 402:
            print("❌ Remove.bg quota exceeded")
            return JSONResponse(
                status_code=402,
                content={
                    "success": False,
                    "error": "Remove.bg monthly quota exceeded. Upgrade your plan at remove.bg",
                    "image": None
                }
            )
        else:
            error_msg = response.text
            print(f"❌ Remove.bg API error: {error_msg}")
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "success": False,
                    "error": f"Remove.bg API error: {error_msg}",
                    "image": None
                }
            )
        
    except requests.Timeout:
        print("❌ Remove.bg request timeout")
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "Request timeout - image too large or slow connection",
                "image": None
            }
        )
    except Exception as e:
        print(f"❌ Background removal error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "image": None
            }
        )

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze clothing item using Claude AI"""
    try:
        if not client:
            return JSONResponse(
                status_code=400,
                content={"error": "Claude AI not configured"}
            )
        
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        media_type = file.content_type or "image/jpeg"
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
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
                            "text": """Analyze this clothing item and return ONLY a JSON object with these exact fields:
{
  "category": "shirt/pants/shoes/jacket/dress/skirt/accessory",
  "color": "primary color name",
  "style": "casual/formal/sporty/elegant",
  "season": "spring/summer/fall/winter/all-season",
  "description": "brief description"
}"""
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        import json
        analysis = json.loads(response_text)
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        print(f"Clothing analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-mannequin")
async def generate_mannequin(request: dict):
    """Generate mannequin composite"""
    try:
        outfit_items = request.get("items", [])
        style = request.get("style", "realistic")
        
        if not outfit_items:
            raise HTTPException(status_code=400, detail="No items provided")
        
        print(f"🎨 Creating mannequin with {len(outfit_items)} items (style: {style})")
        
        # Create basic composite
        composite = create_outfit_composite(outfit_items)
        
        # Convert to base64
        buffered = io.BytesIO()
        composite.save(buffered, format="PNG")
        composite_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print("✅ Mannequin created successfully!")
        
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{composite_base64}",
            "method": "basic-composite"
        })
        
    except Exception as e:
        print(f"❌ Mannequin error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_outfit_composite(items):
    """Create a basic composite of outfit items on mannequin"""
    width, height = 800, 1200
    
    # White background
    composite = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    # Layer positions for different clothing types
    positions = {
        'shirt': (width//2, height//3),
        'top': (width//2, height//3),
        't-shirt': (width//2, height//3),
        'blouse': (width//2, height//3),
        'sweater': (width//2, height//3),
        
        'pants': (width//2, height//2 + 100),
        'bottom': (width//2, height//2 + 100),
        'jeans': (width//2, height//2 + 100),
        'trousers': (width//2, height//2 + 100),
        'skirt': (width//2, height//2 + 100),
        
        'shoes': (width//2, height - 200),
        'shoe': (width//2, height - 200),
        'sneakers': (width//2, height - 200),
        'boots': (width//2, height - 200),
        
        'jacket': (width//2, height//3 - 50),
        'coat': (width//2, height//3 - 50),
        
        'dress': (width//2, height//2),
    }
    
    for item in items:
        try:
            image_data = item.get('image', '')
            if not image_data:
                print(f"⚠️ No image data for item")
                continue
                
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            
            # Resize to fit (preserve aspect ratio)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Get position based on category
            category = item.get('category', 'shirt').lower().strip()
            pos = positions.get(category, (width//2, height//2))
            
            # Center the image at the position
            paste_pos = (pos[0] - img.width//2, pos[1] - img.height//2)
            
            # Paste onto composite with transparency
            composite.paste(img, paste_pos, img)
            print(f"✅ Added {category} to mannequin")
            
        except Exception as e:
            print(f"⚠️ Error compositing item: {e}")
            continue
    
    return composite

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
