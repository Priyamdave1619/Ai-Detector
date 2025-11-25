# backend/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import numpy as np
import io

app = FastAPI()

# 1. Enable CORS (So React can talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Model (Global variable so it loads only once)
MODEL_PATH = "best_model.keras"
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Helper function to process image
def prepare_image(image_bytes):
    # Open image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case of PNG with transparency)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to model's expected size
    img = img.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Expand dims to create batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess (Scale values to -1 to 1 for Xception)
    # IMPORTANT: This must match the training preprocessing
    return preprocess_input(img_array)

@app.get("/")
def home():
    return {"message": "AI Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read file
        contents = await file.read()
        
        # Preprocess
        processed_image = prepare_image(contents)
        
        # Predict
        prediction = model.predict(processed_image)
        score = float(prediction[0][0]) # Extract single float value
        
        # Interpret result (Threshold 0.5)
        # Based on your training: 1 = AI, 0 = Real (or vice versa depending on class_names order)
        # Usually alphabetical: AI=0, Real=1. 
        # CHECK YOUR COLAB OUTPUT: print(class_names). 
        # Assuming: 0 = AI, 1 = Real (If Colab said ['AI', 'Real'])
        # If Colab said ['Real', 'AI'], flip the logic below.
        
        # LOGIC A (If 0=AI, 1=Real)
        if score > 0.5:
            label = "Real"
            confidence = score
        else:
            label = "AI Generated"
            confidence = 1 - score

        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence * 100:.2f}%",
            "raw_score": score
        }

    except Exception as e:
        return {"error": str(e)}

# To run: uvicorn main:app --reload