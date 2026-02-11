from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic
import os
import requests
from PIL import Image
import io
import base64

# Try to import optional dependencies
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ rembg not available - background removal will use Remove.bg API only")

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("⚠️ replicate not available - mannequin generation will use basic compositing")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@app.get("/")
def read_root():
    return {
        "message": "Wardrobe AI Backend is running! 🎨👔",
        "features": {
            "claude_ai": bool(ANTHROPIC_API_KEY),
            "removebg": bool(REMOVEBG_API_KEY),
            "rembg_local": REMBG_AVAILABLE,
            "replicate": REPLICATE_AVAILABLE
        }
    }

@app.post("/analyze-clothing")
async def analyze_clothing(file: UploadFile = File(...)):
    """Analyze a clothing item using Claude AI"""
    try:
        # Read the image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Determine media type
        media_type = file.content_type or "image/jpeg"
        
        # Call Claude API
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
        
        # Parse response
        response_text = message.content[0].text
        # Try to extract JSON if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        import json
        analysis = json.loads(response_text)
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-outfit")
async def generate_outfit(request: dict):
    """Generate outfit suggestions using Claude AI"""
    try:
        wardrobe = request.get("wardrobe", [])
        occasion = request.get("occasion", "casual day out")
        weather = request.get("weather", "mild")
        
        # Build wardrobe description
        wardrobe_text = "\n".join([
            f"- {item['category']}: {item['color']} {item['style']} ({item['season']})"
            for item in wardrobe
        ])
        
        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Given this wardrobe:
{wardrobe_text}

Create 3 outfit combinations for: {occasion} in {weather} weather.

Return ONLY a JSON object with this structure:
{{
  "outfits": [
    {{
      "name": "outfit name",
      "items": ["item1 id", "item2 id", "item3 id"],
      "description": "why this works",
      "style_tips": "styling advice"
    }}
  ]
}}

Use the exact item descriptions from the wardrobe above."""
                }
            ],
        )
        
        # Parse response
        response_text = message.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        import json
        outfits = json.loads(response_text)
        
        return JSONResponse(content=outfits)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from image"""
    try:
        contents = await file.read()
        
        # Try Remove.bg API first (primary method)
        if REMOVEBG_API_KEY:
            try:
                response = requests.post(
                    'https://api.remove.bg/v1.0/removebg',
                    files={'image_file': contents},
                    data={'size': 'auto'},
                    headers={'X-Api-Key': REMOVEBG_API_KEY},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_base64 = base64.b64encode(response.content).decode('utf-8')
                    return JSONResponse(content={
                        "success": True,
                        "image": f"data:image/png;base64,{result_base64}",
                        "method": "removebg"
                    })
            except Exception as e:
                print(f"Remove.bg failed: {e}")
        
        # Fallback to local rembg (if available)
        if REMBG_AVAILABLE:
            try:
                input_image = Image.open(io.BytesIO(contents))
                output_image = remove(input_image)
                
                # Convert to base64
                buffered = io.BytesIO()
                output_image.save(buffered, format="PNG")
                result_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return JSONResponse(content={
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "method": "rembg"
                })
            except Exception as e:
                print(f"Local rembg failed: {e}")
        
        # If both fail, return original image
        result_base64 = base64.b64encode(contents).decode('utf-8')
        return JSONResponse(content={
            "success": False,
            "image": f"data:image/jpeg;base64,{result_base64}",
            "method": "none",
            "message": "Background removal not available - showing original image"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-mannequin")
async def generate_mannequin(request: dict):
    """Generate a dressed mannequin using AI compositing"""
    try:
        outfit_items = request.get("items", [])  # List of {image, category} objects
        mannequin_style = request.get("style", "realistic")
        
        if not outfit_items:
            raise HTTPException(status_code=400, detail="No outfit items provided")
        
        # Create a composite image
        composite = create_outfit_composite(outfit_items)
        
        # Convert to base64
        buffered = io.BytesIO()
        composite.save(buffered, format="PNG")
        composite_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Use Replicate to enhance (if available)
        if REPLICATE_AVAILABLE and REPLICATE_API_TOKEN:
            try:
                output = replicate.run(
                    "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                    input={
                        "image": f"data:image/png;base64,{composite_base64}",
                        "prompt": f"professional fashion photography, {mannequin_style} mannequin wearing stylish outfit, clean white background, studio lighting",
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5
                    }
                )
                
                result_url = output[0] if isinstance(output, list) else output
                result_response = requests.get(result_url)
                result_base64 = base64.b64encode(result_response.content).decode('utf-8')
                
                return JSONResponse(content={
                    "success": True,
                    "image": f"data:image/png;base64,{result_base64}",
                    "method": "replicate-enhanced"
                })
                
            except Exception as e:
                print(f"Replicate enhancement failed: {e}")
        
        # Return basic composite
        return JSONResponse(content={
            "success": True,
            "image": f"data:image/png;base64,{composite_base64}",
            "method": "basic-composite"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_outfit_composite(items):
    """Create a basic composite of outfit items"""
    width, height = 800, 1200
    composite = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    
    positions = {
        'shirt': (width//2, height//3),
        'pants': (width//2, height//2 + 100),
        'shoes': (width//2, height - 200),
        'jacket': (width//2, height//3 - 50),
        'dress': (width//2, height//2),
    }
    
    for item in items:
        try:
            image_data = item.get('image', '')
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            category = item.get('category', 'shirt').lower()
            pos = positions.get(category, (width//2, height//2))
            paste_pos = (pos[0] - img.width//2, pos[1] - img.height//2)
            
            composite.paste(img, paste_pos, img)
            
        except Exception as e:
            print(f"Error compositing item: {e}")
            continue
    
    return composite

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
