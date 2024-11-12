import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { AlertCircle, Clock } from "lucide-react";

// COCO classes that YOLOv5 can detect
const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

const ObjectDetectionApp = () => {
  const webcamRef = useRef(null);
  const [confidence, setConfidence] = useState(0.5);
  const [detections, setDetections] = useState({
    classIDs: [],
    confidences: [],
    boxes: [],
  });
  const [lastUpdate, setLastUpdate] = useState(null);
  const [error, setError] = useState(null);

  const detectObjects = async () => {
    if (webcamRef.current?.video?.readyState !== 4) return;

    try {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(webcamRef.current.video, 0, 0, canvas.width, canvas.height);

      const dataURL = canvas.toDataURL("image/jpeg", 0.8);
      const imageData = dataURL.split(",")[1];

      const response = await axios.post(
        "http://localhost:5000/detect_objects",
        {
          image: imageData,
        }
      );

      setDetections(response.data);
      setLastUpdate(new Date());
      setError(null);
    } catch (error) {
      console.error("Detection error:", error);
      setError(
        "Failed to process detection. Please check if the backend server is running."
      );
    }
  };

  useEffect(() => {
    const interval = setInterval(detectObjects, 2000);
    return () => clearInterval(interval);
  }, [confidence]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow-sm">
            <h1 className="text-2xl font-medium text-gray-900">
              Real-time Object Detection
            </h1>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Clock className="w-4 h-4" />
              <span>
                Last update:{" "}
                {lastUpdate
                  ? new Date(lastUpdate).toLocaleTimeString()
                  : "Never"}
              </span>
            </div>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Camera Feed */}
            <div className="lg:col-span-3">
              <div className="bg-white p-4 rounded-lg shadow-sm">
                <div className="relative rounded-lg overflow-hidden border border-gray-200">
                  <Webcam
                    ref={webcamRef}
                    width={640}
                    height={480}
                    style={{ transform: "scaleX(-1)" }}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>

            {/* Controls and Detection List */}
            <div className="lg:col-span-2 space-y-6">
              {/* Confidence Slider */}
              <div className="bg-white rounded-lg shadow-sm p-4">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-sm font-medium text-gray-700">
                    Confidence Threshold
                  </label>
                  <span className="text-sm font-medium text-blue-600">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500"
                />
              </div>

              {/* Detection List */}
              <div className="bg-white rounded-lg shadow-sm p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-3">
                  Latest Detections
                </h3>
                <div className="space-y-2 max-h-[480px] overflow-y-auto pr-2">
                  {detections.classIDs.length > 0 ? (
                    detections.classIDs
                      .map((classId, index) => {
                        if (detections.confidences[index] >= confidence) {
                          return (
                            <div
                              key={index}
                              className="flex items-center justify-between p-3 bg-gray-50 rounded-md transition-colors hover:bg-gray-100"
                            >
                              <div className="flex items-center gap-3">
                                <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                                <span className="text-sm text-gray-700">
                                  {COCO_CLASSES[classId]}
                                </span>
                              </div>
                              <span className="text-sm font-medium text-blue-600">
                                {(detections.confidences[index] * 100).toFixed(
                                  0
                                )}
                                %
                              </span>
                            </div>
                          );
                        }
                        return null;
                      })
                      .filter(Boolean)
                  ) : (
                    <div className="text-sm text-gray-500 text-center py-4">
                      No objects detected
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionApp;
