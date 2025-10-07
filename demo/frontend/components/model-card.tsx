"use client"

import { Card } from "@/components/ui/card"
import type { Model } from "@/app/page"
import { cn } from "@/lib/utils"

type ModelCardProps = {
  model: Model
  selected: boolean
  onClick: () => void
}

export function ModelCard({ model, selected, onClick }: ModelCardProps) {
  return (
    <Card
      onClick={onClick}
      className={cn(
        "relative cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-xl",
        "bg-slate-800/50 border-slate-700 backdrop-blur-sm",
        "hover:bg-slate-800/80 hover:border-blue-500/50",
        selected && "border-blue-500 shadow-lg shadow-blue-500/20 scale-105",
      )}
    >
      <div className="p-6 space-y-3">
        <div className="w-full h-40 sm:h-48 md:h-56 bg-slate-700/50 rounded-lg flex items-center justify-center overflow-hidden">
          {model.thumbnailUrl ? (
            // Use the server-hosted thumbnail. Keep height fixed, width dynamic so small images scale properly.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={model.thumbnailUrl}
              alt={`${model.name} thumbnail`}
              className="h-full w-auto max-w-full object-contain"
              onError={(e) => {
                // fallback to placeholder if image fails to load
                const target = e.currentTarget as HTMLImageElement
                target.src = '/placeholder.png'
              }}
            />
          ) : (
            <div className="text-slate-500">
              <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                />
              </svg>
            </div>
          )}

          {/* Prefetch weights url as data attribute for analysis component */}
          {model.weightsUrl && <meta data-weights-url={model.weightsUrl} />}

        </div>
        <div className="space-y-1">
          <h3 className="font-medium text-white text-sm">{model.name}</h3>
          {model.confidence && (
            <p className="text-xs text-slate-400">Confidence: {(model.confidence * 100).toFixed(0)}%</p>
          )}
        </div>
      </div>
      {selected && (
        <div className="absolute inset-0 rounded-lg border-2 border-blue-500 pointer-events-none animate-pulse" />
      )}
    </Card>
  )
}
