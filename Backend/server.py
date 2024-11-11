import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/detect_objects": {  # Specify route
        "origins": ["http://localhost:3000"],  # Allow only your frontend
        "methods": ["POST"],  # Allow only POST method
        "allow_headers": ["Content-Type"]
    }
})

# Load YOLO model
net = cv2.dnn.readNetFromONNX('models/yolov5m.onnx')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Get image data from request
    img_data = request.get_json()['image']
    
    # Decode image
    img_array = np.frombuffer(bytes.fromhex(img_data), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Prepare image for detection
    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), 
                                swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform detection
    outputs = net.forward()
    
    # Process results
    boxes = []
    confidences = []
    classIDs = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5:
                box = detection[:4] * np.array([W, H, W, H])
                (x, y, width, height) = box.astype("int")
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    return jsonify({
        'boxes': boxes,
        'confidences': confidences,
        'classIDs': classIDs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)