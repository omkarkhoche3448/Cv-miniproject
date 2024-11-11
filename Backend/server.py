import cv2
import numpy as np
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromONNX('models/yolov5s.onnx')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Get the image data from the request
    img_data = request.get_json()['image']

    # Decode the image data and convert it to a NumPy array
    img_array = np.frombuffer(bytes.fromhex(img_data), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Perform object detection
    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    # Process the detection results
    boxes = []
    confidences = []
    classIDs = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                # Scale the bounding box coordinates back to the original image size
                box = detection[:4] * np.array([W, H, W, H])
                (x, y, width, height) = box.astype("int")
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Return the detection results as JSON
    return jsonify({
        'boxes': boxes,
        'confidences': confidences,
        'classIDs': classIDs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)