# 🚗 YOLO Pothole Detection API

A production-ready computer vision API that detects potholes in road images using YOLOv8, deployed on AWS ECS with FastAPI.

## 🌐 Live Demo

**API Endpoint:** `http://13.220.63.113:8000`  
**Interactive Docs:** [http://13.220.63.113:8000/docs](http://13.220.63.113:8000/docs)

> Try uploading a road image to detect potholes in real-time!

## ✨ Features

- 🎯 **Real-time pothole detection** using custom-trained YOLOv8 model
- 🚀 **FastAPI** for high-performance REST API
- 🐳 **Dockerized** for consistent deployment
- ☁️ **AWS ECS deployment** with auto-scaling capabilities
- 📊 **ONNX Runtime** for optimized inference
- 📝 **Auto-generated API documentation** with Swagger UI
- 🔒 **Production-ready** with proper error handling and logging

## 🏗️ Architecture

```
User → API Gateway → AWS ECS (Fargate) → Docker Container
                                          ├── FastAPI Server
                                          ├── YOLO Model (ONNX)
                                          └── ONNX Runtime
```

## 🛠️ Tech Stack

- **Framework:** FastAPI 0.104.1
- **ML Model:** YOLOv8 (exported to ONNX)
- **Inference:** ONNX Runtime 1.17.0
- **Containerization:** Docker
- **Cloud Platform:** AWS (ECR + ECS Fargate)
- **Image Processing:** Pillow, NumPy

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker
- AWS CLI (for deployment)
- Your trained YOLO model (`my_yolo_model.onnx`)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/yolo-pothole-detector.git
   cd yolo-pothole-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your ONNX model**
   ```bash
   # Place your my_yolo_model.onnx file in the project root
   # Note: Model file is not included in repo due to size
   ```

4. **Run locally**
   ```bash
   uvicorn yolo_pothole_api:app --reload
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

## 🐳 Docker Deployment

### Build the Docker image

```bash
docker build -t yolo-pothole-api:v1 .
```

### Run the container

```bash
docker run -p 8000:8000 yolo-pothole-api:v1
```

## ☁️ AWS Deployment

### 1. Push to ECR

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag yolo-pothole-api:v1 YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yolo-pothole-detector:latest

# Push image
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/yolo-pothole-detector:latest
```

### 2. Deploy to ECS

```bash
# Create cluster
aws ecs create-cluster --cluster-name yolo-cluster --region us-east-1

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1

# Create service
aws ecs create-service \
  --cluster yolo-cluster \
  --service-name yolo-pothole-service \
  --task-definition yolo-pothole-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[YOUR_SUBNET],securityGroups=[YOUR_SG],assignPublicIp=ENABLED}" \
  --region us-east-1
```

## 📡 API Usage

### Health Check

```bash
curl http://13.220.63.113:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "model_file": "my_yolo_model.onnx"
}
```

### Predict Potholes

```bash
curl -X POST "http://13.220.63.113:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@road_image.jpg" \
  -F "confidence_threshold=0.5"
```

**Response:**
```json
{
  "success": true,
  "image_size": {
    "width": 640,
    "height": 480
  },
  "num_potholes_detected": 3,
  "detections": [
    {
      "bbox": {
        "x1": 120.5,
        "y1": 230.2,
        "x2": 245.8,
        "y2": 310.6
      },
      "confidence": 0.87,
      "class": "pothole"
    }
  ]
}
```

## 📊 Model Information

- **Model Type:** YOLOv8 (custom trained)
- **Input Size:** 640x640
- **Format:** ONNX
- **Classes:** Pothole
- **Training Dataset:** Custom pothole dataset

## 🎯 Use Cases

- 🛣️ Road maintenance and monitoring
- 🚗 Autonomous vehicle safety systems
- 📱 Mobile apps for reporting road damage
- 🏛️ Municipal infrastructure management
- 📊 Road quality assessment and analytics

## 🔒 Security Considerations

- API deployed with proper AWS security groups
- Input validation for image uploads
- Rate limiting recommended for production
- Model file excluded from repository (proprietary)

## 📈 Performance

- **Inference Time:** ~100-200ms per image
- **Memory Usage:** ~1.5GB
- **CPU:** 1 vCPU (AWS Fargate)
- **Throughput:** ~10-20 requests/second

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/yunusajib)
- LinkedIn: [Your LinkedIn](linkedin.com/in/yunusajibrin)
- Portfolio: [yourwebsite.com]( yunusajib.github.io/my-portfolio)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- FastAPI by Sebastián Ramírez
- AWS for cloud infrastructure

## 📞 Contact

For questions or feedback, please open an issue or contact me at yunusajib01@gmail.com

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐