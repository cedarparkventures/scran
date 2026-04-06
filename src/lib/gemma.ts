export type ContentMode = 'photo' | 'text' | 'mixed' | 'auto'
export type GemmaStatus = 'idle' | 'downloading' | 'ready' | 'error'

export type ScanResult = {
  text: string
  mode: ContentMode
  modelId: string
}

type ModelState = {
  status: GemmaStatus
  progress: number
  error?: string
}

type PendingLoad = {
  resolve: () => void
  reject: (error: Error) => void
  onProgress?: (status: string, percent: number) => void
}

type PendingScan = {
  resolve: (result: ScanResult) => void
  reject: (error: Error) => void
  onProgress: (status: string, percent: number) => void
}

const modelState: ModelState = { status: 'idle', progress: 0 }

let workerPromise: Promise<Worker> | null = null
let loadPromise: Promise<void> | null = null
const pendingLoads: PendingLoad[] = []
let pendingScan: PendingScan | null = null

function toError(e: unknown) {
  return e instanceof Error ? e : new Error(String(e))
}

function resolvePendingLoads() {
  while (pendingLoads.length) pendingLoads.shift()?.resolve()
}

function rejectPendingLoads(error: Error) {
  while (pendingLoads.length) pendingLoads.shift()?.reject(error)
}

async function getWorker() {
  if (!workerPromise) {
    workerPromise = Promise.resolve(
      new Worker(new URL('./gemma.worker.ts', import.meta.url), { type: 'module' }),
    )

    const worker = await workerPromise

    worker.onmessage = (event: MessageEvent<any>) => {
      const { status, data, progress } = event.data ?? {}

      if (status === 'loading') {
        modelState.status = 'downloading'
        modelState.error = undefined
        pendingLoads.forEach((p) => p.onProgress?.(String(data), modelState.progress))
        pendingScan?.onProgress(String(data), Math.max(modelState.progress, 5))
        return
      }

      if (status === 'progress') {
        modelState.status = 'downloading'
        modelState.progress = Number(progress ?? 0)
        pendingLoads.forEach((p) => p.onProgress?.('Loading Gemma 4', modelState.progress))
        pendingScan?.onProgress('Loading Gemma 4', Math.min(modelState.progress, 30))
        return
      }

      if (status === 'ready') {
        modelState.status = 'ready'
        modelState.progress = 100
        modelState.error = undefined
        resolvePendingLoads()
        return
      }

      if (status === 'scan-status') {
        pendingScan?.onProgress(String(data), Number(progress ?? 35))
        return
      }

      if (status === 'scan-complete') {
        modelState.status = 'ready'
        const current = pendingScan
        pendingScan = null
        current?.resolve({
          text: String(data?.text ?? ''),
          mode: data?.mode ?? 'auto',
          modelId: data?.modelId ?? '',
        })
        return
      }

      if (status === 'error') {
        const err = toError(data ?? 'Gemma 4 failed')
        modelState.status = 'error'
        modelState.progress = 0
        modelState.error = err.message
        rejectPendingLoads(err)
        if (pendingScan) {
          const current = pendingScan
          pendingScan = null
          current.reject(err)
        }
      }
    }

    worker.onerror = (event) => {
      const err = new Error(event.message || 'Gemma 4 worker crashed')
      modelState.status = 'error'
      modelState.progress = 0
      modelState.error = err.message
      rejectPendingLoads(err)
      if (pendingScan) {
        const current = pendingScan
        pendingScan = null
        current.reject(err)
      }
    }
  }

  return workerPromise
}

export function getModelState() {
  return { ...modelState }
}

export function checkWebGpu(): { ok: boolean; message: string } {
  const has = typeof navigator !== 'undefined' && 'gpu' in navigator
  return {
    ok: has,
    message: has
      ? 'WebGPU available — Gemma 4 model downloads on first use'
      : 'WebGPU not available in this browser. Use Chrome or Edge with WebGPU enabled.',
  }
}

export async function initModel(onProgress?: (status: string, percent: number) => void) {
  if (modelState.status === 'ready') {
    onProgress?.('Model ready', 100)
    return
  }

  if (loadPromise) {
    await loadPromise
    onProgress?.('Model ready', 100)
    return
  }

  const worker = await getWorker()
  modelState.status = 'downloading'
  modelState.progress = 0
  modelState.error = undefined

  loadPromise = new Promise<void>((resolve, reject) => {
    pendingLoads.push({ resolve, reject, onProgress })
    worker.postMessage({ type: 'load' })
  }).finally(() => {
    loadPromise = null
  })

  await loadPromise
}

export async function scanPage(
  imageDataUrl: string,
  mode: ContentMode,
  onProgress: (status: string, progress: number) => void,
): Promise<ScanResult> {
  const gpu = checkWebGpu()
  if (!gpu.ok) throw new Error(gpu.message)

  if (pendingScan) throw new Error('A scan is already in progress')

  const worker = await getWorker()

  return new Promise<ScanResult>((resolve, reject) => {
    pendingScan = { resolve, reject, onProgress }
    worker.postMessage({ type: 'scan', imageDataUrl, mode })
  }).catch((error) => {
    throw new Error(`Scan failed: ${toError(error).message}`)
  })
}
