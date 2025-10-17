"use client"

import { useEffect, useRef } from "react"
import { Card } from "@/components/ui/card"

type NeuralNetworkGraphProps = {
  isProcessing: boolean
}

type Node = {
  x: number
  y: number
  radius: number
  activation: number
  layer: number
}

type Edge = {
  from: Node
  to: Node
  weight: number
}

export function NeuralNetworkGraph({ isProcessing }: NeuralNetworkGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)
  const nodesRef = useRef<Node[]>([])
  const edgesRef = useRef<Edge[]>([])
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size
    const updateSize = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
    }
    updateSize()
    window.addEventListener("resize", updateSize)

    // Initialize network structure
    const initNetwork = () => {
      const layers = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 2, 2] // Network architecture
      const nodes: Node[] = []
      const edges: Edge[] = []

      const width = canvas.getBoundingClientRect().width
      const height = canvas.getBoundingClientRect().height
      const layerSpacing = width / (layers.length + 1)

      // Create nodes
      layers.forEach((nodeCount, layerIndex) => {
        const nodeSpacing = height / (nodeCount + 1)
        for (let i = 0; i < nodeCount; i++) {
          nodes.push({
            x: layerSpacing * (layerIndex + 1),
            y: nodeSpacing * (i + 1),
            radius: 8,
            activation: 0,
            layer: layerIndex,
          })
        }
      })

      // Create edges
      let nodeIndex = 0
      for (let layer = 0; layer < layers.length - 1; layer++) {
        const currentLayerSize = layers[layer]
        const nextLayerSize = layers[layer + 1]
        const currentLayerStart = nodeIndex
        const nextLayerStart = nodeIndex + currentLayerSize

        for (let i = 0; i < currentLayerSize; i++) {
          for (let j = 0; j < nextLayerSize; j++) {
            edges.push({
              from: nodes[currentLayerStart + i],
              to: nodes[nextLayerStart + j],
              weight: Math.random(),
            })
          }
        }
        nodeIndex += currentLayerSize
      }

      nodesRef.current = nodes
      edgesRef.current = edges
    }

    initNetwork()

    // Animation loop
    const animate = () => {
      const rect = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, rect.width, rect.height)

      timeRef.current += 0.02

      // Update activations during processing
      if (isProcessing) {
        nodesRef.current.forEach((node) => {
          const wave = Math.sin(timeRef.current * 2 + node.layer * 0.5)
          node.activation = (wave + 1) / 2
        })
      } else {
        // Idle state - gentle pulsing
        nodesRef.current.forEach((node) => {
          const pulse = Math.sin(timeRef.current + node.x * 0.01 + node.y * 0.01)
          node.activation = 0.3 + pulse * 0.1
        })
      }

      // Draw edges
      edgesRef.current.forEach((edge) => {
        const opacity = isProcessing ? edge.from.activation * 0.4 : 0.1
        ctx.beginPath()
        ctx.moveTo(edge.from.x, edge.from.y)
        ctx.lineTo(edge.to.x, edge.to.y)
        ctx.strokeStyle = `rgba(59, 130, 246, ${opacity})`
        ctx.lineWidth = 1
        ctx.stroke()
      })

      // Draw nodes
      nodesRef.current.forEach((node) => {
        // Outer glow
        if (isProcessing && node.activation > 0.5) {
          const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 2)
          gradient.addColorStop(0, `rgba(59, 130, 246, ${node.activation * 0.3})`)
          gradient.addColorStop(1, "rgba(59, 130, 246, 0)")
          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.radius * 2, 0, Math.PI * 2)
          ctx.fill()
        }

        // Node circle
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
        const intensity = node.activation
        ctx.fillStyle = `rgba(59, 130, 246, ${0.3 + intensity * 0.7})`
        ctx.fill()
        ctx.strokeStyle = `rgba(147, 197, 253, ${0.5 + intensity * 0.5})`
        ctx.lineWidth = 2
        ctx.stroke()
      })

  // store the animation id (or null)
  animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
      }
      window.removeEventListener("resize", updateSize)
    }
  }, [isProcessing])

  return (
    <div className="h-full space-y-4">
      <h3 className="text-lg font-semibold text-white">Neural Network Graph</h3>
      <Card className="bg-slate-800/50 border-slate-700 aspect-[4/3] overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" style={{ display: "block" }} />
      </Card>
    </div>
  )
}
