import { useState, useRef, useCallback } from 'react'
import { checkWebGpu, initModel, scanPage, getModelState } from './lib/gemma'
import type { ContentMode, ScanResult } from './lib/gemma'

type ScanEntry = {
  id: string
  fileName: string
  thumbnail: string
  mode: ContentMode
  result: ScanResult | null
  error: string | null
  scanning: boolean
  progress: number
  statusText: string
}

function parseResult(result: ScanResult) {
  const raw = result.text
  const sepIndex = raw.indexOf('---')

  if (result.mode === 'photo' || sepIndex === -1) {
    return { description: raw, fullText: null, summary: null }
  }

  const body = raw.slice(0, sepIndex).trim()
  const summary = raw.slice(sepIndex + 3).trim() || null

  if (result.mode === 'mixed') {
    const visualMatch = body.match(/## Visual Elements\s*\n([\s\S]*?)(?=## Text Content|$)/i)
    const textMatch = body.match(/## Text Content\s*\n([\s\S]*)/i)
    return {
      description: visualMatch?.[1]?.trim() || null,
      fullText: textMatch?.[1]?.trim() || body,
      summary,
    }
  }

  return { description: null, fullText: body, summary }
}

const MODE_OPTIONS: { value: ContentMode; label: string; desc: string }[] = [
  { value: 'auto', label: 'Auto Detect', desc: 'Automatically detect content type' },
  { value: 'photo', label: 'Photo', desc: 'Describe the image' },
  { value: 'text', label: 'Text / OCR', desc: 'Extract and transcribe text' },
  { value: 'mixed', label: 'Mixed', desc: 'Text + images (textbooks, posters)' },
]

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }}
      className="copy-btn"
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

export default function App() {
  const [entries, setEntries] = useState<ScanEntry[]>([])
  const [mode, setMode] = useState<ContentMode>('auto')
  const [modelReady, setModelReady] = useState(false)
  const [modelLoading, setModelLoading] = useState(false)
  const [modelProgress, setModelProgress] = useState('')
  const [modelPercent, setModelPercent] = useState(0)
  const [gpuOk] = useState(() => checkWebGpu())
  const [modeOpen, setModeOpen] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)

  const handleLoadModel = useCallback(async () => {
    if (modelReady || modelLoading) return
    setModelLoading(true)
    try {
      await initModel((status, percent) => {
        setModelProgress(status)
        setModelPercent(percent)
      })
      setModelReady(true)
    } catch (e) {
      setModelProgress(`Error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setModelLoading(false)
    }
  }, [modelReady, modelLoading])

  const readFileAsDataUrl = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = () => reject(new Error('Failed to read file'))
      reader.readAsDataURL(file)
    })

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files?.length) return

    const state = getModelState()
    if (state.status !== 'ready') {
      setModelLoading(true)
      try {
        await initModel((status, percent) => {
          setModelProgress(status)
          setModelPercent(percent)
        })
        setModelReady(true)
      } catch (e) {
        setModelProgress(`Error: ${e instanceof Error ? e.message : String(e)}`)
        setModelLoading(false)
        return
      }
      setModelLoading(false)
    }

    for (const file of Array.from(files)) {
      const dataUrl = await readFileAsDataUrl(file)
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      const entry: ScanEntry = {
        id,
        fileName: file.name,
        thumbnail: dataUrl,
        mode,
        result: null,
        error: null,
        scanning: true,
        progress: 0,
        statusText: 'Starting scan...',
      }

      setEntries((prev) => [entry, ...prev])

      try {
        const result = await scanPage(dataUrl, mode, (status, progress) => {
          setEntries((prev) =>
            prev.map((e) => (e.id === id ? { ...e, statusText: status, progress } : e)),
          )
        })
        setEntries((prev) =>
          prev.map((e) =>
            e.id === id ? { ...e, scanning: false, result, progress: 100, statusText: 'Done' } : e,
          ),
        )
      } catch (err) {
        setEntries((prev) =>
          prev.map((e) =>
            e.id === id
              ? { ...e, scanning: false, error: err instanceof Error ? err.message : String(err) }
              : e,
          ),
        )
      }
    }
  }, [mode])

  const removeEntry = (id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id))
  }

  const selectedMode = MODE_OPTIONS.find((m) => m.value === mode)!

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-icon">S</div>
            <div>
              <h1 className="brand-title">Scran</h1>
              <p className="brand-sub">AI Page Scanner &mdash; Gemma 4 + WebGPU</p>
            </div>
          </div>
          <div className="model-status">
            {!modelReady && !modelLoading && gpuOk.ok && (
              <button onClick={handleLoadModel} className="btn btn-primary">
                Load Model
              </button>
            )}
            {modelLoading && (
              <span className="status-loading">
                <span className="spinner" />
                {modelProgress} ({modelPercent}%)
              </span>
            )}
            {modelReady && (
              <span className="status-ready">
                <span className="dot" />
                Ready
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        {!gpuOk.ok && (
          <div className="alert alert-error">{gpuOk.message}</div>
        )}

        {/* Controls */}
        <div className="controls">
          <div className="mode-selector">
            <button className="btn btn-mode" onClick={() => setModeOpen(!modeOpen)}>
              {selectedMode.label} ▾
            </button>
            {modeOpen && (
              <>
                <div className="backdrop" onClick={() => setModeOpen(false)} />
                <div className="dropdown">
                  {MODE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => { setMode(opt.value); setModeOpen(false) }}
                      className={`dropdown-item ${mode === opt.value ? 'active' : ''}`}
                    >
                      <div className="dropdown-label">{opt.label}</div>
                      <div className="dropdown-desc">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!gpuOk.ok}
            className="btn btn-primary"
          >
            Upload Image
          </button>
          <button
            onClick={() => cameraInputRef.current?.click()}
            disabled={!gpuOk.ok}
            className="btn btn-secondary"
          >
            Take Photo
          </button>

          <input ref={fileInputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => handleFiles(e.target.files)} />
          <input ref={cameraInputRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
        </div>

        {/* Drop zone */}
        {entries.length === 0 && (
          <div
            className="dropzone"
            onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add('dropzone-active') }}
            onDragLeave={(e) => e.currentTarget.classList.remove('dropzone-active')}
            onDrop={(e) => {
              e.preventDefault()
              e.currentTarget.classList.remove('dropzone-active')
              handleFiles(e.dataTransfer.files)
            }}
          >
            <div className="dropzone-icon">↑</div>
            <p className="dropzone-text">Drop images here or use the buttons above</p>
            <p className="dropzone-sub">Photos, documents, textbook pages &mdash; any image</p>
          </div>
        )}

        {/* Results */}
        <div className="entries">
          {entries.map((entry) => {
            const parsed = entry.result ? parseResult(entry.result) : null
            return (
              <div key={entry.id} className="card">
                <div className="card-header">
                  <img src={entry.thumbnail} alt="" className="card-thumb" />
                  <div className="card-meta">
                    <p className="card-filename">{entry.fileName}</p>
                    <p className="card-mode">Mode: {MODE_OPTIONS.find((m) => m.value === entry.mode)?.label}</p>
                  </div>
                  {entry.scanning && (
                    <span className="status-loading">
                      <span className="spinner" />
                      {entry.statusText} ({entry.progress}%)
                    </span>
                  )}
                  <button onClick={() => removeEntry(entry.id)} className="btn-close">✕</button>
                </div>

                {entry.scanning && (
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${entry.progress}%` }} />
                  </div>
                )}

                {entry.error && <div className="alert alert-error">{entry.error}</div>}

                {parsed && (
                  <div className="card-body">
                    <div className="card-image">
                      <img src={entry.thumbnail} alt={entry.fileName} />
                    </div>

                    {parsed.description && (
                      <div className="result-section">
                        <div className="result-header">
                          <h3 className="result-title title-indigo">
                            {entry.mode === 'mixed' ? 'Visual Elements' : 'Description'}
                          </h3>
                          <CopyButton text={parsed.description} />
                        </div>
                        <p className="result-text">{parsed.description}</p>
                      </div>
                    )}

                    {parsed.fullText && (
                      <div className="result-section">
                        <div className="result-header">
                          <h3 className="result-title title-green">Full Text</h3>
                          <CopyButton text={parsed.fullText} />
                        </div>
                        <pre className="result-pre">{parsed.fullText}</pre>
                      </div>
                    )}

                    {parsed.summary && (
                      <div className="result-section">
                        <div className="result-header">
                          <h3 className="result-title title-amber">Summary</h3>
                          <CopyButton text={parsed.summary} />
                        </div>
                        <p className="result-summary">{parsed.summary}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </main>
    </div>
  )
}
