import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# Page configuration
st.set_page_config(
    page_title="Pothole Detector",
    page_icon="🚗",
    layout="wide"
)

# API endpoint
API_URL = "http://100.25.40.78:8000"


# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🚗 AI Pothole Detection System")
st.markdown("### Detect road potholes in real-time using YOLOv8")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=YOLO+Detector",
             use_column_width=True)
    st.markdown("## ⚙️ Settings")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="IoU threshold for removing duplicate detections (lower = more strict)"
    )

    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This app uses a custom-trained YOLOv8 model deployed on AWS ECS to detect potholes in road images.
    
    **Tech Stack:**
    - YOLOv8 (ONNX)
    - FastAPI
    - Docker
    - AWS ECS
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown(
        "[GitHub Repo](https://github.com/yunusajib/yolo-pothole-detector)")
    st.markdown("[API Docs](http://13.220.63.113:8000/docs)")
    st.markdown("[Portfolio](https://yunusajib.github.io)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a road image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a road"
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

with col2:
    st.markdown("## 🎯 Detection Results")

    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing image..."):
            try:
                # Prepare the request
                # Include filename and content type for proper file upload
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                params = {
                    "confidence_threshold": confidence_threshold,
                    "iou_threshold": iou_threshold
                }

                # Call API
                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    params=params,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display metrics
                    st.success(f"✅ Analysis Complete!")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Potholes Found",
                                  result["num_potholes_detected"])
                    with metric_col2:
                        st.metric("Image Width",
                                  f"{result['image_size']['width']}px")
                    with metric_col3:
                        st.metric("Image Height",
                                  f"{result['image_size']['height']}px")

                    # Draw bounding boxes
                    if result["num_potholes_detected"] > 0:
                        image_with_boxes = image.copy()
                        draw = ImageDraw.Draw(image_with_boxes)

                        # Try to use a nice font, fall back to default if not available
                        try:
                            font = ImageFont.truetype("Arial.ttf", 20)
                        except:
                            font = ImageFont.load_default()

                        for i, detection in enumerate(result["detections"]):
                            bbox = detection["bbox"]
                            confidence = detection["confidence"]

                            # Draw rectangle
                            draw.rectangle(
                                [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                                outline="red",
                                width=3
                            )

                            # Draw label
                            label = f"Pothole {i+1}: {confidence:.2%}"
                            draw.text(
                                (bbox["x1"], bbox["y1"] - 25),
                                label,
                                fill="red",
                                font=font
                            )

                        st.image(
                            image_with_boxes, caption="Detected Potholes", use_column_width=True)

                        # Show detection details
                        st.markdown("### 📋 Detection Details")
                        for i, detection in enumerate(result["detections"]):
                            with st.expander(f"Pothole {i+1} - Confidence: {detection['confidence']:.2%}"):
                                st.json({
                                    "Bounding Box": detection["bbox"],
                                    "Confidence": f"{detection['confidence']:.4f}",
                                    "Class": detection["class"]
                                })
                    else:
                        st.info(
                            "No potholes detected in this image. Try adjusting the confidence threshold or upload a different image.")
                        st.image(image, caption="No Detections",
                                 use_column_width=True)

                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out. The API might be starting up. Please try again in a moment.")
            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Could not connect to API. Please check if the service is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload an image to get started!")
        st.markdown("""
        ### 💡 Tips for Best Results:
        - Use clear, well-lit images
        - Make sure potholes are visible
        - Avoid blurry or low-quality images
        - Road should be the main subject
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>Built with ❤️ by Yunusa Jibrin | Powered by YOLOv8 + FastAPI + AWS</p>
    <p><a href='https://github.com/yunusajib' style='color: white;'>GitHub</a> | 
       <a href='https://linkedin.com/in/yunusajibrin' style='color: white;'>LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
