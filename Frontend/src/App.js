import React, { useState, useRef, useEffect } from 'react';
import { AlertCircle, Play, Square, Camera } from 'lucide-react';
import Webcam from 'react-webcam';

const RELEVANT_CLASSES = {
  0: "person",
  1: "bicycle",
  2: "car",
  3: "motorcycle",
  5: "bus",
  6: "train",
  7: "truck",
  67: "cell phone",
  62: "tv",
  63: "laptop",
  64: "mouse",
  65: "remote",
  66: "keyboard",
};

const ObjectDetectionApp = () => {
  const webcamRef = useRef(null);
  const [detections, setDetections] = useState(new Map()); // Using Map to track unique detections
  const [error, setError] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const detectionInterval = useRef(null);
  const [uniqueDetectionCount, setUniqueDetectionCount] = useState(0);

  const startDetection = () => {
    if (!isCameraReady) return;
    setDetections(new Map());
    setUniqueDetectionCount(0);
    setIsDetecting(true);
    detectionInterval.current = setInterval(detectObjects, 1000);
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (detectionInterval.current) {
      clearInterval(detectionInterval.current);
      detectionInterval.current = null;
    }
  };

  const isNewDetection = (detection, existingDetections) => {
    const key = `${detection.class_id}-${Math.round(detection.box[0]/50)}-${Math.round(detection.box[1]/50)}`;
    if (!existingDetections.has(key)) {
      return key;
    }
    return null;
  };

  const detectObjects = async () => {
    if (!webcamRef.current?.video?.readyState === 4) return;

    try {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(webcamRef.current.video, 0, 0, canvas.width, canvas.height);
      
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
      const formData = new FormData();
      formData.append('image', blob);

      const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const newDetections = await response.json();
      
      // Process only new detections
      setDetections(prevDetections => {
        const updatedDetections = new Map(prevDetections);
        let newCount = 0;

        newDetections.forEach(detection => {
          const detectionKey = isNewDetection(detection, prevDetections);
          if (detectionKey && detection.confidence >= confidenceThreshold) {
            updatedDetections.set(detectionKey, {
              ...detection,
              timestamp: Date.now(),
              id: detectionKey
            });
            newCount++;
          }
        });

        setUniqueDetectionCount(prev => prev + newCount);
        return updatedDetections;
      });

      setError(null);
    } catch (err) {
      console.error('Detection error:', err);
      setError('Failed to process detection. Please check if the backend server is running.');
      stopDetection();
    }
  };

  useEffect(() => {
    return () => {
      if (detectionInterval.current) {
        clearInterval(detectionInterval.current);
      }
    };
  }, []);

  const handleUserMedia = () => {
    setIsCameraReady(true);
  };

  const resetDetections = () => {
    setDetections(new Map());
    setUniqueDetectionCount(0);
  };

  // Convert Map to Array for rendering
  const detectionsArray = Array.from(detections.values());

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold text-gray-900">Single-Detection Object Recognition</h1>
            <div className="flex items-center space-x-4">
              {!isDetecting ? (
                <button
                  onClick={startDetection}
                  disabled={!isCameraReady}
                  className={`flex items-center px-4 py-2 rounded-md ${
                    isCameraReady 
                      ? 'bg-blue-500 text-white hover:bg-blue-600' 
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Detection
                </button>
              ) : (
                <button
                  onClick={stopDetection}
                  className="flex items-center px-4 py-2 rounded-md bg-red-500 text-white hover:bg-red-600"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop Detection
                </button>
              )}
              <button
                onClick={resetDetections}
                className="flex items-center px-4 py-2 rounded-md bg-gray-500 text-white hover:bg-gray-600"
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-lg">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Camera Feed */}
          <div className="lg:col-span-3">
            <div className="bg-white p-4 rounded-lg shadow-sm">
              <div className="relative rounded-lg overflow-hidden border border-gray-200">
                <Webcam
                  ref={webcamRef}
                  width={640}
                  height={480}
                  onUserMedia={handleUserMedia}
                  style={{ transform: 'scaleX(-1)' }}
                  className="w-full h-full object-cover"
                />
                <svg className="absolute top-0 left-0 w-full h-full" style={{ transform: 'scaleX(-1)' }}>
                  {isDetecting && detectionsArray.map((detection) => (
                    <g key={detection.id}>
                      <rect
                        x={detection.box[0]}
                        y={detection.box[1]}
                        width={detection.box[2]}
                        height={detection.box[3]}
                        fill="none"
                        stroke="#00ff00"
                        strokeWidth="2"
                      />
                      <text
                        x={detection.box[0]}
                        y={detection.box[1] - 5}
                        fill="#00ff00"
                        fontSize="12px"
                      >
                        {RELEVANT_CLASSES[detection.class_id]}
                        {' '}
                        {(detection.confidence * 100).toFixed(0)}%
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            </div>
          </div>

          {/* Controls and Detection List */}
          <div className="lg:col-span-2 space-y-6">
            {/* Stats */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <div className="text-sm font-medium text-gray-700">
                Unique Objects Detected: {uniqueDetectionCount}
              </div>
            </div>

            {/* Confidence Slider */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-gray-700">
                  Confidence Threshold
                </label>
                <span className="text-sm font-medium text-blue-600">
                  {(confidenceThreshold * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-gray-200"
              />
            </div>

            {/* Detection List */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">
                Detected Objects
              </h3>
              <div className="space-y-2 max-h-[480px] overflow-y-auto pr-2">
                {detectionsArray.length > 0 ? (
                  detectionsArray
                    .sort((a, b) => b.timestamp - a.timestamp)
                    .map((detection) => (
                      <div
                        key={detection.id}
                        className="flex items-center justify-between p-3 bg-gray-50 rounded-md hover:bg-gray-100"
                      >
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-green-500" />
                          <span className="text-sm text-gray-700">
                            {RELEVANT_CLASSES[detection.class_id]}
                          </span>
                        </div>
                        <span className="text-sm font-medium text-blue-600">
                          {(detection.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))
                ) : (
                  <div className="text-sm text-gray-500 text-center py-4">
                    {isDetecting ? 'No objects detected yet' : 'Detection stopped'}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionApp;