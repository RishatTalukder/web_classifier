import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'

const App = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  
  const [loading, setLoading] = useState(true)
  const [model, setModel] = useState<tf.GraphModel | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<string | null>(null);

  // 1. Load the Graph Model
  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadGraphModel('model/model.json')

        tf.tidy(() => {
          loadedModel.predict(tf.zeros([1, 28, 28, 1]))
        })

        setModel(loadedModel)
        setLoading(false)
        console.log('model loaded')
      } catch (e) {
        console.error(e)
      }
    }
    loadModel()
  }, [])

  // 2. Initialize Canvas (Black background, white ink)
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = 280;
    canvas.height = 280;

    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) return;

    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "white";
    context.lineWidth = 16; // Thick stroke matches MNIST data properties

    contextRef.current = context;
  }, [loading]); // Re-run once loading completes and canvas hits DOM

  // 3. Drawing Interactions
  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    const { offsetX, offsetY } = getCoords(e);
    contextRef.current?.beginPath();
    contextRef.current?.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !contextRef.current) return;
    e.preventDefault();
    const { offsetX, offsetY } = getCoords(e);
    contextRef.current.lineTo(offsetX, offsetY);
    contextRef.current.stroke();

    // Trigger prediction while drawing
    runInference();
  };

  const stopDrawing = () => {
    contextRef.current?.closePath();
    setIsDrawing(false);
  };

  const getCoords = (e: React.MouseEvent | React.TouchEvent) => {
    if ('touches' in e.nativeEvent) {
      const rect = canvasRef.current!.getBoundingClientRect();
      return {
        offsetX: e.nativeEvent.touches[0].clientX - rect.left,
        offsetY: e.nativeEvent.touches[0].clientY - rect.top
      };
    }
    const mouseEvent = e as React.MouseEvent;
    return { offsetX: mouseEvent.nativeEvent.offsetX, offsetY: mouseEvent.nativeEvent.offsetY };
  };

  const clearCanvas = () => {
    if (!canvasRef.current || !contextRef.current) return;
    contextRef.current.fillStyle = "black";
    contextRef.current.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setPrediction(null);
    setConfidence(null);
  };

  // 4. Real-time Inference Engine
  const runInference = () => {
    if (!model || !canvasRef.current) return;

    tf.tidy(() => {
      // Get 1-channel grayscale pixel values
      const tensor = tf.browser.fromPixels(canvasRef.current!, 1);
      const resized = tf.image.resizeBilinear(tensor, [28, 28]);
      const normalized = resized.div(tf.scalar(255.0));
      const inputBatch = normalized.expandDims(0); // Structuring matrix data shape to [1, 28, 28, 1]

      const output = model.predict(inputBatch) as tf.Tensor;
      const scores = output.dataSync();
      const highestScoreIndex = output.argMax(-1).dataSync()[0];

      setPrediction(highestScoreIndex);
      setConfidence((scores[highestScoreIndex] * 100).toFixed(1));
    });
  };

  return (
    <div style={{ textAlign: 'center', fontFamily: 'sans-serif', padding: '20px' }}>
      <h1>Life is a Race</h1>
      {loading ? <p>Loading model...</p> : <p>✅ Model Ready! Draw below.</p>}

      {!loading && (
        <>
          <div style={{ display: 'inline-block', border: '2px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
            <canvas
              ref={canvasRef}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
              style={{ cursor: 'crosshair', display: 'block', touchAction: 'none' }}
            />
          </div>

          <div style={{ margin: '15px' }}>
            <button onClick={clearCanvas} style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>
              Clear Canvas
            </button>
          </div>

          {prediction !== null && (
            <div style={{ marginTop: '20px', fontSize: '24px' }}>
              <strong>Prediction: {prediction}</strong>
              <div style={{ fontSize: '16px', color: '#666', marginTop: '5px' }}>
                Confidence: {confidence}%
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default App
