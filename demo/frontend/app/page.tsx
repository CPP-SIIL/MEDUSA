"use client"

import { useState, useEffect } from "react"
import { ModelSelectionPanel } from "@/components/model-selection-panel"
import { AnalysisInterface } from "@/components/analysis-interface"

export type Model = {
  id: string
  name: string
  type: "positive" | "negative"
  confidence?: number
  // URLs served by the backend (http://localhost:5000)
  thumbnailUrl?: string
  weightsUrl?: string
  // server-side model path (e.g. /models/positive/xxx.stl) used for inference
  url?: string
}

export type PredictionResult = {
  class: "positive" | "negative"
  confidence: number
  probabilities: {
    positive: number
    negative: number
  }
}

// We'll fetch available demo models from the backend at runtime.
// The backend serves JSON from http://localhost:5000/api/getModels
// Example model shape returned by backend: { name, type, thumbnail, weights, confidence }


export default function STLClassificationDemo() {
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [positiveModels, setPositiveModels] = useState<Model[]>([])
  const [negativeModels, setNegativeModels] = useState<Model[]>([])

  useEffect(() => {
    // Fetch model list from backend
    async function loadModels() {
      try {
        const res = await fetch('http://127.0.0.1:5000/api/getModels')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()

        // backend returns { models: [{ name, type, thumbnail, weights, confidence }] }
        const positives: Model[] = []
        const negatives: Model[] = []

        for (const m of data.models || []) {
          const model: Model = {
            id: m.name,
            name: m.name,
            type: m.type,
            confidence: m.confidence,
            thumbnailUrl: m.thumbnail ? `http://127.0.0.1:5000${m.thumbnail}` : undefined,
            weightsUrl: m.weights ? `http://127.0.0.1:5000${m.weights}` : undefined,
            // backend returns a url field pointing to the model/file served by the server
            // keep the raw server-side path (e.g. "/models/positive/xxx.stl") so the server can match it
            url: m.url ? m.url : undefined,
          }
          if (m.type === 'positive') positives.push(model)
          else negatives.push(model)
        }

        setPositiveModels(positives)
        setNegativeModels(negatives)
      } catch (err) {
        // fallback: leave arrays empty; UI will show placeholders
        console.error('Failed to load models from backend', err)
      }
    }

    loadModels()
  }, [])

  const handleModelSelect = async (model: Model) => {
    setSelectedModel(model)
    setIsProcessing(true)
    setPredictionResult(null)

    // If backend model URL not provided, fall back to simulated result
    if (!model.url) {
      console.warn('No model.url available from backend; falling back to simulated result.')
      setTimeout(() => {
        const result: PredictionResult = {
          class: model.type,
          confidence: model.confidence || 0.85,
          probabilities: {
            positive: model.type === "positive" ? model.confidence || 0.85 : 1 - (model.confidence || 0.85),
            negative: model.type === "negative" ? model.confidence || 0.85 : 1 - (model.confidence || 0.85),
          },
        }
        setPredictionResult(result)
        setIsProcessing(false)
      }, 1000)
      return
    }

    try {
      const resp = await fetch('http://127.0.0.1:5000/api/doInference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // server expects the server-side path (e.g. "/models/positive/xxx.stl")
        body: JSON.stringify({ modelUrl: model.url })
      })

      if (!resp.ok) {
        const txt = await resp.text()
        throw new Error(`HTTP ${resp.status}: ${txt}`)
      }

      const payload = await resp.json()

      // Expected payload shape (demo server/test_api): { success: true, result: { class, confidence, probabilities } }
      if (!payload || !payload.success || !payload.result) {
        throw new Error(payload?.error || 'Invalid response from inference API')
      }

      const serverResult = payload.result

      // Robust mapping of probabilities and class
      const serverPrediction = serverResult.prediction || serverResult
      const probs = serverPrediction.probabilities
      const predClassRaw = serverPrediction.class

      let positiveProb = 0
      let negativeProb = 0

      if (probs && typeof probs === 'object' && !Array.isArray(probs)) {
        // Named probabilities: prefer explicit keys
        positiveProb = Number(probs.positive ?? probs.pos ?? probs[1] ?? 0)
        negativeProb = Number(probs.negative ?? probs.neg ?? probs[0] ?? 0)
      } else if (Array.isArray(probs) && probs.length >= 2) {
        // Array case: try to respect returned predicted class ordering when possible
        const a = Number(probs[0] ?? 0)
        const b = Number(probs[1] ?? 0)
        if (predClassRaw === 1 || predClassRaw === 'positive') {
          positiveProb = Math.max(a, b)
          negativeProb = Math.min(a, b)
        } else if (predClassRaw === 0 || predClassRaw === 'negative') {
          negativeProb = Math.max(a, b)
          positiveProb = Math.min(a, b)
        } else {
          // Default ordering used by server: [negative, positive]
          negativeProb = a
          positiveProb = b
        }
      } else {
        // Fallback: use confidence + class
        const conf = Number(serverPrediction.confidence ?? serverResult.confidence ?? 0)
        if (predClassRaw === 1 || predClassRaw === 'positive') {
          positiveProb = conf
          negativeProb = 1 - conf
        } else if (predClassRaw === 0 || predClassRaw === 'negative') {
          negativeProb = conf
          positiveProb = 1 - conf
        } else {
          // Give a small positive probability if nothing else is available
          positiveProb = model.type === 'positive' ? 0.5 : 0.5
          negativeProb = 1 - positiveProb
        }
      }

      // Normalize probabilities to sum to 1 (avoid server order issues)
      const sum = positiveProb + negativeProb
      if (sum > 0) {
        positiveProb = Math.min(Math.max(positiveProb / sum, 0), 1)
        negativeProb = Math.min(Math.max(negativeProb / sum, 0), 1)
      } else {
        positiveProb = 0
        negativeProb = 1
      }

      const finalClass =
        predClassRaw === 1 || predClassRaw === 'positive'
          ? 'positive'
          : predClassRaw === 0 || predClassRaw === 'negative'
          ? 'negative'
          : positiveProb >= negativeProb
          ? 'positive'
          : 'negative'

      const mapped: PredictionResult = {
        class: finalClass,
        confidence: Number(serverPrediction.confidence ?? serverResult.confidence ?? Math.max(positiveProb, negativeProb)),
        probabilities: {
          positive: positiveProb,
          negative: negativeProb,
        },
      }

      setPredictionResult(mapped)
    } catch (err) {
      console.error('Inference request failed:', err)
      setPredictionResult(null)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      <div className="mx-auto max-w-[1600px] space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-white">MEDUSA</h1>
          <p className="text-slate-400">Machine-learning Engine for Detecting Unlawful Shapes Automatically</p>
        </div>

        {/* Model Selection Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ModelSelectionPanel
            type="positive"
            models={positiveModels}
            selectedModel={selectedModel}
            onSelect={handleModelSelect}
          />
          <ModelSelectionPanel
            type="negative"
            models={negativeModels}
            selectedModel={selectedModel}
            onSelect={handleModelSelect}
          />
        </div>

        {/* Analysis Interface */}
        <AnalysisInterface
          selectedModel={selectedModel}
          isProcessing={isProcessing}
          predictionResult={predictionResult}
        />
      </div>
    </div>
  )
}
