import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app, resources={
    r"/detect_objects": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Load YOLO model
try:
    net = cv2.dnn.readNetFromONNX('models/yolov5m.onnx')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please ensure you have the YOLOv5m ONNX model in the models directory")

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Get image data from request
        img_data = request.get_json()['image']
        
        # Decode base64 image
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Get image dimensions
        (H, W) = img.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Get detections
        outputs = net.forward()[0]
        
        # Initialize lists for results
        boxes = []
        confidences = []
        classIDs = []
        
        # Process detections
        for detection in outputs:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5:
                # Scale bounding box coordinates back relative to image size
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                width = int(detection[2] * W)
                height = int(detection[3] * H)
                
                # Calculate top-left corner of bounding box
                x = max(0, int(center_x - width/2))
                y = max(0, int(center_y - height/2))
                
                # Ensure coordinates are within image bounds
                x = min(x, W)
                y = min(y, H)
                width = min(width, W-x)
                height = min(height, H-y)
                
                # Add detection to results
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                classIDs.append(int(classID))
        
        return jsonify({
            'boxes': boxes,
            'confidences': confidences,
            'classIDs': classIDs
        })
        
    except Exception as e:
        print(f"Error processing detection: {str(e)}")
        return jsonify({
            'boxes': [],
            'confidences': [],
            'classIDs': []
        })

if __name__ == '__main__':
    app.run(debug=True)