from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, resources={
    r"/detect": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST","GET"],
        "allow_headers": ["Content-Type"]
    }
})

# Define relevant class IDs
RELEVANT_CLASSES = {0, 1, 2, 3, 5, 6, 7, 67, 62, 63, 64, 65, 66}

# Initialize rate limiting dictionary
last_processing_time = {}
MIN_PROCESSING_INTERVAL = 0.1  # Minimum time between processing requests (in seconds)

# Store previous detections for tracking
previous_detections = []
DETECTION_THRESHOLD = 0.4  # IOU threshold for considering an object as already detected

# Load YOLO model
try:
    net = cv2.dnn.readNetFromONNX('models/yolov5m.onnx')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please ensure you have the YOLOv5m ONNX model in the models directory")

def clean_old_requests():
    """Clean up old rate limiting entries"""
    global last_processing_time
    current_time = time.time()
    last_processing_time = {
        ip: timestamp 
        for ip, timestamp in last_processing_time.items() 
        if current_time - timestamp < 60
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    # Convert to x1, y1, x2, y2 format
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def is_new_detection(detection, previous_detections):
    """Check if this is a new detection"""
    for prev_detection in previous_detections:
        if (prev_detection['class_id'] == detection['class_id'] and
            calculate_iou(prev_detection['box'], detection['box']) > DETECTION_THRESHOLD):
            return False
    return True

def process_image(image_data):
    """Process image data and return detections"""
    try:
        # Convert image to format suitable for OpenCV
        if isinstance(image_data, str):
            # If image is base64 string
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            # If image is file upload
            image_bytes = image_data.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        # Resize image if too large
        max_size = 1024
        height, width = img.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Get image dimensions
        height, width = img.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0,
            (640, 640),
            swapRB=True,
            crop=False
        )

        # Set input to the model
        net.setInput(blob)

        # Run forward pass
        outputs = net.forward()[0]

        # Process detections
        detections = []
        for detection in outputs:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])

            # Filter by relevant classes and confidence
            if confidence > 0.1 and class_id in RELEVANT_CLASSES:
                # Scale bounding box coordinates back relative to image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate coordinates of the top-left corner
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Create detection object
                detection_obj = {
                    'class_id': int(class_id),
                    'confidence': confidence,
                    'box': [x, y, w, h]
                }

                # Only add if it's a new detection
                if is_new_detection(detection_obj, previous_detections):
                    detections.append(detection_obj)
                    previous_detections.append(detection_obj)

        return detections

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []
@app.route('/detect', methods=['POST', 'GET'])
def detect():
    # Health check endpoint
    if request.method == 'GET':
        return jsonify({
            'status': 'healthy',
            'model_loaded': net is not None
        })

    # Original POST handling
    global last_processing_time
    
    client_ip = request.remote_addr
    
    # Rate limiting check
    current_time = time.time()
    if client_ip in last_processing_time:
        time_since_last = current_time - last_processing_time[client_ip]
        if time_since_last < MIN_PROCESSING_INTERVAL:
            return jsonify({
                'error': 'Too many requests',
                'retry_after': MIN_PROCESSING_INTERVAL - time_since_last,
                'message': 'Please wait before sending another request'
            }), 429

    try:
        # Validate content type
        if not request.content_type or 'multipart/form-data' not in request.content_type:
            return jsonify({
                'error': 'Invalid content type',
                'message': 'Request must be multipart/form-data'
            }), 415

        # Check for image file
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please include an image file in the request'
            }), 400

        image_file = request.files['image']
        
        # Validate file type
        if not image_file.filename or '.' not in image_file.filename:
            return jsonify({
                'error': 'Invalid filename',
                'message': 'Please provide a valid image file'
            }), 400

        # Update last processing time
        last_processing_time[client_ip] = current_time

        # Process image
        detections = process_image(image_file)
        
        # Clean up old rate limiting entries
        clean_old_requests()
        
        return jsonify({
            'status': 'success',
            'detections': detections
        })

    except Exception as e:
        print(f"Error in detect endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_detections():
    """Reset the previous detections list"""
    global previous_detections
    previous_detections = []
    return jsonify({
        'status': 'success',
        'message': 'Detection history cleared'
    })

@app.route('/status', methods=['GET'])
def status():
    """Get API status and statistics"""
    return jsonify({
        'status': 'operational',
        'active_clients': len(last_processing_time),
        'tracked_detections': len(previous_detections),
        'model_loaded': net is not None
    })

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Unhandled error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)