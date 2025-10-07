import { Card } from "@/components/ui/card"
import type { PredictionResult } from "@/app/page"
import { cn } from "@/lib/utils"

type ResultsPanelProps = {
  predictionResult: PredictionResult | null
  isProcessing: boolean
}

export function ResultsPanel({ predictionResult, isProcessing }: ResultsPanelProps) {
  return (
    <div className="h-full space-y-4">
      <h3 className="text-lg font-semibold text-white">Output</h3>
      <Card className="bg-slate-800/50 border-slate-700 p-6 space-y-6">
        {isProcessing ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center space-y-2">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
              <p className="text-sm text-slate-400">Analyzing...</p>
            </div>
          </div>
        ) : predictionResult ? (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Prediction Class */}
            <div className="space-y-2">
              <p className="text-sm text-slate-400">Classification</p>
              <div
                className={cn(
                  "text-2xl font-bold capitalize",
                  predictionResult.class === "positive" ? "text-red-500" : "text-green-500",
                )}
              >
                {predictionResult.class === "positive" ? "Gun Part" : "Non-Gun Part"}
              </div>
            </div>

            {/* Confidence */}
            <div className="space-y-2">
              <p className="text-sm text-slate-400">Confidence</p>
              <div className="space-y-2">
                <div className="text-3xl font-bold text-white">{(predictionResult.confidence * 100).toFixed(1)}%</div>
                <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-1000",
                      predictionResult.confidence > 0.8
                        ? "bg-green-500"
                        : predictionResult.confidence > 0.6
                          ? "bg-yellow-500"
                          : "bg-red-500",
                    )}
                    style={{ width: `${predictionResult.confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Probability Distribution */}
            <div className="space-y-3">
              <p className="text-sm text-slate-400">Probability Distribution</p>
              <div className="space-y-3">
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Positive (Gun Part)</span>
                    <span className="text-slate-400">
                      {(predictionResult.probabilities.positive * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-red-500 rounded-full transition-all duration-1000"
                      style={{ width: `${predictionResult.probabilities.positive * 100}%` }}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Negative (Non-Gun Part)</span>
                    <span className="text-slate-400">
                      {(predictionResult.probabilities.negative * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full bg-green-500 rounded-full transition-all duration-1000"
                      style={{ width: `${predictionResult.probabilities.negative * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <p className="text-sm text-slate-500">Select a model to see results</p>
          </div>
        )}
      </Card>
    </div>
  )
}
