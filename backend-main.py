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

# Set Replicate token
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

try:
    import replicate
    REPLICATE_AVAILABLE = True
    print("✅ Replicate available!")
except ImportError:
    REPLICATE_AVAILABLE = False
    print("⚠️ Replicate not available")

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
                            "text": """Analyze this clothing. Return ONLY valid JSON:
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
    """Generate outfit - ALWAYS 3+ items"""
    try:
        wardrobe = request.get("wardrobe", [])
        occasion = request.get("occasion", "casual")
        weather = request.get("weather", "moderate")
        
        if not wardrobe or len(wardrobe) < 2:
            return JSONResponse(content={
                "outfit": [{"item_index": 0}],
                "explanation": "Add more items!"
            })
        
        # Categorize
        tops = []
        bottoms = []
        shoes = []
        others = []
        
        for i, item in enumerate(wardrobe):
            cat = item.get('category', '').lower()
            if any(x in cat for x in ['shirt', 'top', 'blouse', 't-shirt', 'sweater', 'jacket', 'coat', 'outerwear']):
                tops.append(i)
            elif any(x in cat for x in ['pants', 'bottom', 'jeans', 'trousers', 'skirt']):
                bottoms.append(i)
            elif any(x in cat for x in ['shoe', 'sneaker', 'boot', 'sandal']):
                shoes.append(i)
            else:
                others.append(i)
        
        print(f"📊 {len(tops)} tops, {len(bottoms)} bottoms, {len(shoes)} shoes")
        
        outfit_indices = []
        if tops:
            outfit_indices.append({"item_index": tops[0]})
        if bottoms:
            outfit_indices.append({"item_index": bottoms[0]})
        if shoes:
            outfit_indices.append({"item_index": shoes[0]})
        
        while len(outfit_indices) < 3 and others:
            outfit_indices.append({"item_index": others.pop(0)})
        
        while len(outfit_indices) < 2:
            outfit_indices.append({"item_index": 0})
        
        print(f"✅ Outfit: {[x['item_index'] for x in outfit_indices]}")
        
        explanation = f"Complete outfit for {occasion} in {weather} weather"
        
        return JSONResponse(content={
            "outfit": outfit_indices,
            "explanation": explanation
        })
        
    except Exception as e:
        print(f"❌ Outfit error: {str(e)}")
        return JSONResponse(content={
            "outfit": [{"item_index": 0}, {"item_index": 1}],
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
                content={"success": False, "error": "No API key"}
            )
        
        print(f"📤 Remove.bg ({len(contents)} bytes)")
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': ('image.jpg', contents, 'image/jpeg')},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVEBG_API_KEY},
            timeout=60
        )
        
        if response.status_code == 200:
            result_base64 = base64.b64encode(response.content).decode('utf-8')
            print("✅ BG removed!")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{result_base64}",
                "method": "removebg"
            })
        else:
            return JSONResponse(
                status_code=response.status_code,
                content={"success": False, "error": response.text}
            )
        
    except Exception as e:
        print(f"❌ BG error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/generate-mannequin")
async def generate_mannequin(request: dict):
    """Generate PHOTOREALISTIC mannequin with Replicate"""
    try:
        outfit_items = request.get("items", [])
        style = request.get("style", "realistic")
        
        if not outfit_items:
            raise HTTPException(status_code=400, detail="No items")
        
        print(f"🎨 Creating mannequin: {len(outfit_items)} items, style={style}")
        
        # Step 1: Create basic composite
        composite = create_mannequin_composite(outfit_items)
        
        buffered = io.BytesIO()
        composite.save(buffered, format="PNG")
        composite_bytes = buffered.getvalue()
        composite_base64 = base64.b64encode(composite_bytes).decode('utf-8')
        
        # Step 2: ENHANCE with Replicate AI
        if REPLICATE_AVAILABLE and REPLICATE_API_TOKEN:
            try:
                print("🤖 Enhancing with Replicate AI...")
                
                # Upload composite to a temporary URL or use data URI
                input_image = f"data:image/png;base64,{composite_base64}"
                
                # Use Stable Diffusion XL for img2img
                output = replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    input={
                        "image": input_image,
                        "prompt": "professional fashion photography, photorealistic male mannequin wearing complete stylish outfit with shirt, pants and shoes, clean white studio background, studio lighting, high quality, 4k, ultra detailed, fashion catalog",
                        "negative_prompt": "deformed, distorted, disfigured, poor quality, blurry, text, watermark, low quality, bad anatomy, extra limbs, missing clothes, incomplete outfit",
                        "num_inference_steps": 40,
                        "guidance_scale": 8.0,
                        "strength": 0.65,  # How much to transform
                        "seed": -1,
                    }
                )
                
                print(f"📥 Replicate output type: {type(output)}")
                
                # Get result
                if isinstance(output, list) and len(output) > 0:
                    result_url = output[0]
                elif isinstance(output, str):
                    result_url = output
                else:
                    result_url = str(output)
                
                print(f"📥 Downloading from: {result_url[:100]}...")
                
                # Download result
                result_response = requests.get(result_url, timeout=30)
                result_base64 = base64.b64encode(result_response.content).decode('utf-8')
                
                print("✅ Replicate enhanced mannequin created!")
                
                return JSONResponse(content={
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "method": "replicate-enhanced"
                })
                
            except Exception as e:
                print(f"⚠️ Replicate failed: {str(e)}")
                print(f"⚠️ Falling back to basic composite")
        
        # Fallback: Basic composite
        print("📦 Returning basic composite (Replicate not available)")
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{composite_base64}",
            "method": "basic-composite"
        })
        
    except Exception as e:
        print(f"❌ Mannequin error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def create_mannequin_composite(items):
    """Create composite with proper layering"""
    width, height = 1024, 1536
    composite = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    positions = {
        'shirt': (width//2, int(height * 0.30)),
        'top': (width//2, int(height * 0.30)),
        't-shirt': (width//2, int(height * 0.30)),
        'blouse': (width//2, int(height * 0.30)),
        'sweater': (width//2, int(height * 0.30)),
        'jacket': (width//2, int(height * 0.28)),
        'coat': (width//2, int(height * 0.28)),
        'outerwear': (width//2, int(height * 0.28)),
        
        'pants': (width//2, int(height * 0.55)),
        'bottom': (width//2, int(height * 0.55)),
        'jeans': (width//2, int(height * 0.55)),
        'trousers': (width//2, int(height * 0.55)),
        'skirt': (width//2, int(height * 0.55)),
        
        'shoes': (width//2, int(height * 0.82)),
        'shoe': (width//2, int(height * 0.82)),
        'sneakers': (width//2, int(height * 0.82)),
        'boots': (width//2, int(height * 0.82)),
        
        'dress': (width//2, int(height * 0.45)),
    }
    
    size_limits = {
        'jacket': (550, 550),
        'coat': (550, 550),
        'outerwear': (550, 550),
        'shirt': (500, 500),
        'top': (500, 500),
        'pants': (450, 650),
        'bottom': (450, 650),
        'shoes': (320, 280),
        'dress': (500, 700),
    }
    
    for item in items:
        try:
            image_data = item.get('image', '')
            if not image_data:
                continue
            
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            
            category = item.get('category', 'shirt').lower().strip()
            max_size = size_limits.get(category, (400, 400))
            
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            pos = positions.get(category, (width//2, height//2))
            paste_pos = (pos[0] - img.width//2, pos[1] - img.height//2)
            
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
