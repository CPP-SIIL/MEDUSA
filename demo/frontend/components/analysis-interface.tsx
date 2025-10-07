import { Card } from "@/components/ui/card"
import { ModelPreview } from "@/components/model-preview"
import { NeuralNetworkGraph } from "@/components/neural-network-graph"
import { ResultsPanel } from "@/components/results-panel"
import type { Model, PredictionResult } from "@/app/page"

type AnalysisInterfaceProps = {
  selectedModel: Model | null
  isProcessing: boolean
  predictionResult: PredictionResult | null
}

export function AnalysisInterface({ selectedModel, isProcessing, predictionResult }: AnalysisInterfaceProps) {
  return (
    <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Panel: 3D Model Preview */}
          <div className="lg:col-span-3">
            <ModelPreview selectedModel={selectedModel} isProcessing={isProcessing} />
          </div>

          {/* Center Panel: Neural Network Graph */}
          <div className="lg:col-span-6">
            <NeuralNetworkGraph isProcessing={isProcessing} />
          </div>

          {/* Right Panel: Output Results */}
          <div className="lg:col-span-3">
            <ResultsPanel predictionResult={predictionResult} isProcessing={isProcessing} />
          </div>
        </div>
      </div>
    </Card>
  )
}
