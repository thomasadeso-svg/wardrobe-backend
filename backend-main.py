from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
import requests
from PIL import Image
import io
import base64
import json

app = FastAPI()

# CORS
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

# Try to import replicate
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("⚠️ Replicate not installed")

@app.get("/")
def read_root():
    return {
        "message": "Wardrobe AI Backend 🎨👔",
        "status": "online",
        "features": {
            "claude_ai": bool(ANTHROPIC_API_KEY),
            "removebg": bool(REMOVEBG_API_KEY),
            "replicate": REPLICATE_AVAILABLE and bool(REPLICATE_API_TOKEN)
        }
    }

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze clothing with Claude Vision"""
    try:
        if not client:
            return JSONResponse(content={
                "category": "top",
                "color": "blue",
                "style": "casual",
                "season": "all-season",
                "description": "Clothing item"
            })
        
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        media_type = file.content_type or "image/jpeg"
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
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
                            "text": """Analyze this clothing item. Return ONLY valid JSON (no markdown):
{
  "category": "shirt/pants/shoes/jacket/dress/skirt",
  "color": "main color",
  "style": "casual/formal/sporty",
  "season": "spring/summer/fall/winter/all-season",
  "description": "brief description"
}"""
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        print(f"✅ Analysis: {result}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return JSONResponse(content={
            "category": "top",
            "color": "unknown",
            "style": "casual",
            "season": "all-season",
            "description": "Clothing item"
        })

@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    """Generate outfit with Claude"""
    try:
        wardrobe = request.get("wardrobe", [])
        occasion = request.get("occasion", "casual")
        weather = request.get("weather", "moderate")
        
        if not wardrobe or len(wardrobe) < 2:
            return JSONResponse(content={
                "outfit": [{"item_index": 0}],
                "explanation": "Add more items!"
            })
        
        if not client:
            return JSONResponse(content={
                "outfit": [{"item_index": 0}, {"item_index": 1}],
                "explanation": "Random outfit (AI offline)"
            })
        
        wardrobe_text = "\n".join([
            f"{i}. {item.get('category', 'item')}: {item.get('color', 'colored')} {item.get('style', 'style')}"
            for i, item in enumerate(wardrobe)
        ])
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Wardrobe:
{wardrobe_text}

Create outfit for {occasion} in {weather} weather.

Return ONLY JSON (no markdown):
{{
  "outfit": [{{"item_index": 0}}, {{"item_index": 1}}, {{"item_index": 2}}],
  "explanation": "why this works"
}}

Use indices 0-{len(wardrobe)-1}. Include top, bottom, and shoes if available."""
            }]
        )
        
        response_text = message.content[0].text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ Outfit error: {str(e)}")
        return JSONResponse(content={
            "outfit": [{"item_index": 0}, {"item_index": min(1, len(wardrobe)-1)}],
            "explanation": "Outfit suggestion"
        })

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background with Remove.bg"""
    try:
        contents = await file.read()
        
        if not REMOVEBG_API_KEY:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No API key", "image": None}
            )
        
        print(f"📤 Remove.bg request ({len(contents)} bytes)")
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': ('image.jpg', contents, 'image/jpeg')},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVEBG_API_KEY},
            timeout=60
        )
        
        print(f"📥 Remove.bg status: {response.status_code}")
        
        if response.status_code == 200:
            result_base64 = base64.b64encode(response.content).decode('utf-8')
            print("✅ Background removed!")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{result_base64}",
                "method": "removebg"
            })
        else:
            print(f"❌ Remove.bg error: {response.status_code}")
            return JSONResponse(
                status_code=response.status_code,
                content={"success": False, "error": response.text, "image": None}
            )
        
    except Exception as e:
        print(f"❌ BG removal error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "image": None}
        )

@app.post("/generate-mannequin")
async def generate_mannequin(request: dict):
    """Generate photorealistic mannequin"""
    try:
        outfit_items = request.get("items", [])
        style = request.get("style", "realistic")
        
        if not outfit_items:
            raise HTTPException(status_code=400, detail="No items")
        
        print(f"🎨 Creating mannequin ({len(outfit_items)} items, style: {style})")
        
        # Create composite
        composite = create_mannequin_composite(outfit_items)
        
        # Convert to base64
        buffered = io.BytesIO()
        composite.save(buffered, format="PNG")
        composite_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Try Replicate enhancement
        if REPLICATE_AVAILABLE and REPLICATE_API_TOKEN:
            try:
                print("🤖 Enhancing with Replicate AI...")
                
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={
                        "image": f"data:image/png;base64,{composite_base64}",
                        "prompt": f"professional fashion photography, {style} mannequin wearing complete outfit with shirt, pants and shoes, clean white studio background, professional lighting, high quality, 4k",
                        "negative_prompt": "deformed, distorted, disfigured, poor details, bad anatomy, incomplete outfit, missing items",
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                        "strength": 0.6
                    }
                )
                
                result_url = output[0] if isinstance(output, list) else output
                result_response = requests.get(result_url, timeout=30)
                result_base64 = base64.b64encode(result_response.content).decode('utf-8')
                
                print("✅ Replicate enhanced!")
                return JSONResponse(content={
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "method": "replicate-enhanced"
                })
                
            except Exception as e:
                print(f"⚠️ Replicate failed: {e}")
        
        # Return basic composite
        print("✅ Basic composite created")
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{composite_base64}",
            "method": "basic-composite"
        })
        
    except Exception as e:
        print(f"❌ Mannequin error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_mannequin_composite(items):
    """Create mannequin composite with PROPER positioning"""
    width, height = 1024, 1536  # Larger canvas
    
    # White background
    composite = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    # IMPROVED POSITIONS - More space for shoes!
    positions = {
        # Tops (higher, more centered)
        'shirt': (width//2, int(height * 0.30)),
        'top': (width//2, int(height * 0.30)),
        't-shirt': (width//2, int(height * 0.30)),
        'blouse': (width//2, int(height * 0.30)),
        'sweater': (width//2, int(height * 0.30)),
        'jacket': (width//2, int(height * 0.28)),
        'coat': (width//2, int(height * 0.28)),
        
        # Bottoms (middle)
        'pants': (width//2, int(height * 0.55)),
        'bottom': (width//2, int(height * 0.55)),
        'jeans': (width//2, int(height * 0.55)),
        'trousers': (width//2, int(height * 0.55)),
        'skirt': (width//2, int(height * 0.55)),
        
        # SHOES (lower, but visible!)
        'shoes': (width//2, int(height * 0.82)),
        'shoe': (width//2, int(height * 0.82)),
        'sneakers': (width//2, int(height * 0.82)),
        'boots': (width//2, int(height * 0.82)),
        'sandals': (width//2, int(height * 0.82)),
        
        # Dresses (full body)
        'dress': (width//2, int(height * 0.45)),
    }
    
    # Size limits for each category
    size_limits = {
        'shirt': (500, 500),
        'top': (500, 500),
        'jacket': (550, 550),
        'pants': (450, 600),
        'bottom': (450, 600),
        'shoes': (300, 250),  # Smaller shoes
        'dress': (500, 700),
    }
    
    for item in items:
        try:
            image_data = item.get('image', '')
            if not image_data:
                continue
            
            # Decode base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            
            # Get category
            category = item.get('category', 'shirt').lower().strip()
            
            # Get size limit
            max_size = size_limits.get(category, (400, 400))
            
            # Resize with aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Get position
            pos = positions.get(category, (width//2, height//2))
            
            # Center at position
            paste_pos = (pos[0] - img.width//2, pos[1] - img.height//2)
            
            # Paste with transparency
            composite.paste(img, paste_pos, img)
            print(f"✅ Added {category} at {paste_pos}")
            
        except Exception as e:
            print(f"⚠️ Error adding item: {e}")
            continue
    
    return composite

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
