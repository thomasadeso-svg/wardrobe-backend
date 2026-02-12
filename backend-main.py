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
import time

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

if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Import replicate
REPLICATE_AVAILABLE = False
if REPLICATE_API_TOKEN:
    try:
        import replicate
        REPLICATE_AVAILABLE = True
        print("✅ Replicate loaded successfully!")
    except ImportError:
        print("⚠️ Replicate not installed - run: pip install replicate")
else:
    print("⚠️ No REPLICATE_API_TOKEN found")

@app.get("/")
def read_root():
    return {
        "message": "Wardrobe AI Backend 🎨👔",
        "status": "online",
        "features": {
            "claude_ai": bool(ANTHROPIC_API_KEY),
            "removebg": bool(REMOVEBG_API_KEY),
            "replicate": REPLICATE_AVAILABLE
        }
    }

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    try:
        if not client:
            return JSONResponse(content={
                "category": "top", "color": "blue", "style": "casual",
                "season": "all-season", "description": "Clothing item"
            })
        
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        media_type = file.content_type or "image/jpeg"
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_image}},
                    {"type": "text", "text": 'Analyze clothing. Return JSON: {"category": "shirt/pants/shoes/jacket", "color": "color", "style": "casual/formal", "season": "spring/summer/fall/winter", "description": "text"}'}
                ],
            }],
        )
        
        response_text = message.content[0].text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return JSONResponse(content={"category": "top", "color": "unknown", "style": "casual", "season": "all-season", "description": ""})

@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    try:
        wardrobe = request.get("wardrobe", [])
        
        if len(wardrobe) < 2:
            return JSONResponse(content={"outfit": [{"item_index": 0}], "explanation": "Add more items!"})
        
        # Categorize wardrobe
        tops, bottoms, shoes, others = [], [], [], []
        
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
        
        print(f"📊 Wardrobe: {len(tops)} tops, {len(bottoms)} bottoms, {len(shoes)} shoes, {len(others)} other")
        
        # Build outfit (one from each category)
        outfit_indices = []
        if tops: outfit_indices.append({"item_index": tops[0]})
        if bottoms: outfit_indices.append({"item_index": bottoms[0]})
        if shoes: outfit_indices.append({"item_index": shoes[0]})
        
        # Fill to at least 2 items
        while len(outfit_indices) < 2 and others:
            outfit_indices.append({"item_index": others.pop(0)})
        if len(outfit_indices) == 0:
            outfit_indices = [{"item_index": 0}]
        
        print(f"✅ Generated outfit: {[x['item_index'] for x in outfit_indices]}")
        
        return JSONResponse(content={
            "outfit": outfit_indices,
            "explanation": f"Complete outfit for {request.get('occasion', 'casual')}"
        })
    except Exception as e:
        print(f"❌ Outfit error: {e}")
        return JSONResponse(content={"outfit": [{"item_index": 0}], "explanation": "Error"})

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        if not REMOVEBG_API_KEY:
            return JSONResponse(status_code=400, content={"success": False, "error": "No API key"})
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': ('image.jpg', contents, 'image/jpeg')},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVEBG_API_KEY},
            timeout=60
        )
        
        if response.status_code == 200:
            result_base64 = base64.b64encode(response.content).decode('utf-8')
            return JSONResponse(content={"success": True, "image": f"data:image/png;base64,{result_base64}", "method": "removebg"})
        else:
            return JSONResponse(status_code=response.status_code, content={"success": False, "error": response.text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/generate-mannequin")
async def generate_mannequin(request: dict):
    """Generate mannequin - REALISTIC uses Replicate AI!"""
    try:
        outfit_items = request.get("items", [])
        style = request.get("style", "minimal")  # realistic, minimal, artistic
        
        if not outfit_items:
            raise HTTPException(status_code=400, detail="No items")
        
        print(f"🎨 Mannequin request: {len(outfit_items)} items, style={style}")
        
        # Create composite
        composite = create_mannequin_composite(outfit_items)
        buffered = io.BytesIO()
        composite.save(buffered, format="PNG")
        composite_bytes = buffered.getvalue()
        composite_base64 = base64.b64encode(composite_bytes).decode('utf-8')
        
        # MINIMAL = Just return composite
        if style == "minimal":
            print("📦 Returning minimal composite")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{composite_base64}",
                "method": "minimal"
            })
        
        # REALISTIC or ARTISTIC = Use Replicate
        if not REPLICATE_AVAILABLE:
            print("⚠️ Replicate not available, returning composite")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{composite_base64}",
                "method": "basic-composite"
            })
        
        try:
            print(f"🤖 Enhancing with Replicate ({style})...")
            
            # Different prompts for different styles
            if style == "realistic":
                prompt = "professional fashion photography, photorealistic male mannequin wearing complete stylish outfit with all clothing items visible, clean white studio background, professional studio lighting, high quality, ultra detailed, 4k, fashion catalog style"
                negative = "deformed, distorted, disfigured, low quality, blurry, incomplete outfit, missing items, text, watermark"
                strength = 0.70
            else:  # artistic
                prompt = "artistic fashion illustration, stylized mannequin wearing trendy outfit, minimalist aesthetic, clean lines, modern art style, white background"
                negative = "photorealistic, deformed, distorted, low quality"
                strength = 0.75
            
            # Call Replicate API
            output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "image": f"data:image/png;base64,{composite_base64}",
                    "prompt": prompt,
                    "negative_prompt": negative,
                    "num_inference_steps": 35,
                    "guidance_scale": 7.5,
                    "strength": strength,
                }
            )
            
            print(f"📥 Replicate output received")
            
            # Get URL
            if isinstance(output, list):
                result_url = output[0]
            else:
                result_url = str(output)
            
            print(f"📥 Downloading result...")
            
            # Download
            result_response = requests.get(result_url, timeout=60)
            result_base64 = base64.b64encode(result_response.content).decode('utf-8')
            
            print(f"✅ Replicate {style} mannequin created!")
            
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{result_base64}",
                "method": f"replicate-{style}"
            })
            
        except Exception as e:
            print(f"⚠️ Replicate error: {e}")
            print(f"⚠️ Falling back to composite")
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{composite_base64}",
                "method": "basic-composite"
            })
        
    except Exception as e:
        print(f"❌ Mannequin error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def create_mannequin_composite(items):
    """Create composite with ALL items properly positioned"""
    width, height = 1024, 1536
    composite = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    # Positions (percentage of height)
    positions = {
        'outerwear': (width//2, int(height * 0.28)),
        'jacket': (width//2, int(height * 0.28)),
        'coat': (width//2, int(height * 0.28)),
        'shirt': (width//2, int(height * 0.32)),
        'top': (width//2, int(height * 0.32)),
        't-shirt': (width//2, int(height * 0.32)),
        'blouse': (width//2, int(height * 0.32)),
        'sweater': (width//2, int(height * 0.30)),
        
        'pants': (width//2, int(height * 0.58)),
        'bottom': (width//2, int(height * 0.58)),
        'jeans': (width//2, int(height * 0.58)),
        'trousers': (width//2, int(height * 0.58)),
        'skirt': (width//2, int(height * 0.58)),
        
        'shoes': (width//2, int(height * 0.85)),
        'shoe': (width//2, int(height * 0.85)),
        'sneakers': (width//2, int(height * 0.85)),
        'boots': (width//2, int(height * 0.85)),
        'sandals': (width//2, int(height * 0.85)),
        
        'dress': (width//2, int(height * 0.45)),
    }
    
    # Sizes
    sizes = {
        'outerwear': (580, 600),
        'jacket': (550, 580),
        'shirt': (500, 500),
        'top': (500, 500),
        'pants': (480, 700),
        'bottom': (480, 700),
        'shoes': (350, 300),
        'dress': (520, 750),
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
            max_size = sizes.get(category, (400, 400))
            
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            pos = positions.get(category, (width//2, height//2))
            paste_pos = (pos[0] - img.width//2, pos[1] - img.height//2)
            
            composite.paste(img, paste_pos, img)
            print(f"✅ Layered {category}")
            
        except Exception as e:
            print(f"⚠️ Item error: {e}")
    
    return composite

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
