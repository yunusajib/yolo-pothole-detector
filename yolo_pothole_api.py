# FastAPI YOLO Pothole Detection API
# Production-ready computer vision API for portfolio!

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
from typing import List, Dict

# ========================================
# STEP 1: Load YOLO Model
# ========================================
print("🚀 Loading YOLO pothole detection model...")

# Load ONNX model
session = None
error_message = None

try:
    onnx_model_path = "my_yolo_model.onnx"

    # Check if file exists
    if not os.path.exists(onnx_model_path):
        error_message = f"Model file not found at: {onnx_model_path}"
        print(f"❌ {error_message}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in directory: {os.listdir('.')}")
    else:
        print(f"📁 Model file found: {onnx_model_path}")
        print(
            f"   File size: {os.path.getsize(onnx_model_path) / 1024 / 1024:.2f} MB")

        session = ort.InferenceSession(onnx_model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Input name: {session.get_inputs()[0].name}")
        print(f"   Input shape: {session.get_inputs()[0].shape}")
        print(f"   Output name: {session.get_outputs()[0].name}")
except Exception as e:
    error_message = str(e)
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    session = None

# ========================================
# STEP 2: Create FastAPI app
# ========================================
app = FastAPI(
    title="YOLO Pothole Detection API",
    description="Detects potholes in road images using YOLO",
    version="1.0.0"
)

# ========================================
# STEP 3: Helper Functions
# ========================================


def preprocess_image(image: Image.Image, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    Preprocess image for YOLO model
    Args:
        image: PIL Image
        target_size: Target size for model (width, height)
    Returns:
        Preprocessed numpy array
    """
    # Resize image
    img_resized = image.resize(target_size)

    # Convert to numpy array
    img_array = np.array(img_resized)

    # Convert RGB to BGR if needed (YOLO often expects BGR)
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Normalize pixel values to 0-1
    img_array = img_array.astype(np.float32) / 255.0

    # Transpose to CHW format (Channel, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def postprocess_predictions(outputs: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
    """
    Process YOLO output to extract detections
    """
    detections = []

    output = outputs[0]

    # Handle YOLO output format: [batch, features, num_predictions]
    # Transpose to [batch, num_predictions, features]
    if len(output.shape) == 3:
        output = np.transpose(output, (0, 2, 1))

    # Process each detection
    for detection in output[0]:
        # Extract bbox coordinates and class scores
        x_center, y_center, width, height = detection[0:4]
        class_scores = detection[4:]

        # Get max confidence
        max_conf = float(np.max(class_scores))

        if max_conf > conf_threshold:
            # Convert from center format to corner format
            x1 = float(x_center - width / 2)
            y1 = float(y_center - height / 2)
            x2 = float(x_center + width / 2)
            y2 = float(y_center + height / 2)

            detections.append({
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "confidence": max_conf,
                "class": "pothole"
            })

    return detections

# ========================================
# STEP 4: API Endpoints
# ========================================


@app.get("/")
async def home():
    """Root endpoint with API information"""
    return {
        "message": "🚗 YOLO Pothole Detection API",
        "description": "Upload road images to detect potholes",
        "endpoints": {
            "POST /predict": "Upload an image to detect potholes",
            "GET /health": "Check API health",
            "GET /docs": "Interactive API documentation"
        },
        "model": "YOLOv8 ONNX",
        "author": "Your Name"
    }


@app.get("/health")
async def health_check():
    """Check if API and model are working"""
    model_status = "loaded" if session is not None else "not_loaded"
    response = {
        "status": "healthy" if session else "unhealthy",
        "model": model_status,
        "model_file": "my_yolo_model.onnx"
    }
    if error_message:
        response["error"] = error_message
    return response


@app.post("/predict")
async def predict_potholes(
    file: UploadFile = File(..., description="Road image file"),
    confidence_threshold: float = 0.5
):
    """
    Detect potholes in uploaded road image

    Args:
        file: Image file (JPG, PNG)
        confidence_threshold: Minimum confidence for detections (0.0 to 1.0)

    Returns:
        JSON with detected potholes and their locations
    """
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload JPG or PNG image."
        )

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Get original image size
        original_size = image.size

        # Preprocess for model
        input_tensor = preprocess_image(image, target_size=(640, 640))

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        # Process outputs
        detections = postprocess_predictions(
            outputs, conf_threshold=confidence_threshold)

        # Scale bounding boxes back to original image size
        scale_x = original_size[0] / 640  # Scale factor for width
        scale_y = original_size[1] / 640  # Scale factor for height

        for detection in detections:
            detection["bbox"]["x1"] *= scale_x
            detection["bbox"]["x2"] *= scale_x
            detection["bbox"]["y1"] *= scale_y
            detection["bbox"]["y2"] *= scale_y

        return {
            "success": True,
            "image_size": {
                "width": original_size[0],
                "height": original_size[1]
            },
            "num_potholes_detected": len(detections),
            "detections": detections,
            "confidence_threshold": confidence_threshold
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_base64")
async def predict_from_base64(data: dict):
    """
    Alternative endpoint accepting base64 encoded images
    Useful for web applications
    """
    import base64

    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Process same as above
        input_tensor = preprocess_image(image)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        detections = postprocess_predictions(
            outputs, conf_threshold=data.get("confidence", 0.5))

        return {
            "success": True,
            "num_potholes_detected": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")

# ========================================
# TO RUN THIS:
# ========================================
# 1. Install: pip install fastapi uvicorn onnxruntime pillow numpy opencv-python
# 2. Save your my_yolo_model.onnx in the same folder
# 3. Run: uvicorn yolo_pothole_api:app --reload
# 4. Visit: http://localhost:8000/docs
# 5. Try uploading a road image!
# ========================================
