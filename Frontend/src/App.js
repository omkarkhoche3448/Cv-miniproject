import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const ObjectDetectionApp = () => {
  const webcamRef = useRef(null);
  const [model, setModel] = useState('yolov5s');
  const [confidence, setConfidence] = useState(0.5);
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      detectObjects();
    }, 100);

    return () => clearInterval(interval);
  }, []);

  const detectObjects = async () => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      // Capture the video frame
      const canvas = document.createElement('canvas');
      canvas.width = webcamRef.current.video.width;
      canvas.height = webcamRef.current.video.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(webcamRef.current.video, 0, 0, canvas.width, canvas.height);

      // Convert the canvas to a data URL
      const dataURL = canvas.toDataURL('image/jpeg', 0.8);
      const imageData = dataURL.split(',')[1];

      // Send the image data to the backend for object detection
      const response = await axios.post('/detect_objects', { image: imageData });
      setDetections(response.data);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex justify-center items-center">
      <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-4xl">
        <h1 className="text-3xl font-bold mb-6">Real-Time Object Detection</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <Webcam
              ref={webcamRef}
              width={640}
              height={480}
              style={{ transform: 'scaleX(-1)' }}
              className="rounded-lg shadow-md"
            />
            <canvas id="output-canvas" width={640} height={480} />
          </div>
          <div>
            <div className="mb-6">
              <label htmlFor="model" className="block font-medium mb-2">
                Model:
              </label>
              <select
                id="model"
                className="border border-gray-300 rounded-md py-2 px-3 w-full"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="yolov5s">YOLOv5s</option>
           
              </select>
            </div>
            <div>
              <label htmlFor="confidence" className="block font-medium mb-2">
                Confidence Threshold:
              </label>
              <div className="flex items-center">
                <input
                  id="confidence"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-full appearance-none cursor-pointer"
                />
                <span className="ml-4">{confidence.toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionApp;