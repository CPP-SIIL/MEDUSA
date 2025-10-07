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

  const handleModelSelect = (model: Model) => {
    setSelectedModel(model)
    setIsProcessing(true)
    setPredictionResult(null)

    // Simulate prediction process
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
    }, 3000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      <div className="mx-auto max-w-[1600px] space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-white">STL Classification Demo</h1>
          <p className="text-slate-400">Graph Neural Network for 3D Model Classification</p>
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
