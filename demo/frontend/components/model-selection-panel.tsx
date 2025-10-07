"use client"

import { Card } from "@/components/ui/card"
import { ModelCard } from "@/components/model-card"
import type { Model } from "@/app/page"

type ModelSelectionPanelProps = {
  type: "positive" | "negative"
  models: Model[]
  selectedModel: Model | null
  onSelect: (model: Model) => void
}

export function ModelSelectionPanel({ type, models, selectedModel, onSelect }: ModelSelectionPanelProps) {
  const title = type === "positive" ? "Positive" : "Negative"
  const subtitle = type === "positive" ? "Gun Parts" : "Non-Gun Parts"

  return (
    <Card className="bg-slate-900/50 border-slate-800 backdrop-blur-sm">
      <div className="p-6 space-y-4">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold text-white">{title}</h2>
          <p className="text-sm text-slate-400">{subtitle}</p>
        </div>

        {/* Horizontal scroller */}
        <div className="w-full">
          <div className="flex gap-4 overflow-x-auto py-2 px-1 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-slate-900">
            {models.map((model) => (
              <div key={model.id} className="min-w-[180px] flex-shrink-0">
                <ModelCard
                  model={model}
                  selected={selectedModel?.id === model.id}
                  onClick={() => onSelect(model)}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </Card>
  )
}
