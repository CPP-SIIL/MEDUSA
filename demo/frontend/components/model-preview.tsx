import { Card } from "@/components/ui/card"
import type { Model } from "@/app/page"
import { Spinner } from "@/components/ui/spinner"

type ModelPreviewProps = {
  selectedModel: Model | null
  isProcessing: boolean
}

export function ModelPreview({ selectedModel, isProcessing }: ModelPreviewProps) {
  return (
    <div className="h-full space-y-4">
      <h3 className="text-lg font-semibold text-white">3D Model Preview</h3>
      <Card className="bg-slate-800/50 border-slate-700 aspect-square flex items-center justify-center">
        {isProcessing ? (
          <div className="flex flex-col items-center gap-4">
            <Spinner className="w-8 h-8 text-blue-500" />
            <p className="text-sm text-slate-400">Processing...</p>
          </div>
        ) : selectedModel ? (
          <div className="flex flex-col items-center gap-4 p-6 text-center">
            {/** Show thumbnail if available */}
            {'thumbnailUrl' in selectedModel ? (
              // @ts-ignore
              <img src={selectedModel.thumbnailUrl} alt={selectedModel.name} className="w-40 h-40 object-contain rounded-md bg-black/20" />
            ) : (
              <div className="text-slate-400">
                <svg className="w-20 h-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
                  />
                </svg>
              </div>
            )}
            <div className="space-y-1">
              <p className="text-sm font-medium text-white">{selectedModel.name}</p>
              <p className="text-xs text-slate-400 capitalize">{selectedModel.type}</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-slate-500">Select a model to preview</p>
        )}
      </Card>
    </div>
  )
}
