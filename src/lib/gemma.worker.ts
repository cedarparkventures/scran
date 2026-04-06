import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  env,
  load_image,
} from '@huggingface/transformers'

const MODEL_ID = 'onnx-community/gemma-4-E2B-it-ONNX'
const MAX_SOFT_TOKENS = 1120

type ContentMode = 'photo' | 'text' | 'mixed' | 'auto'

type WorkerRequest =
  | { type: 'load' }
  | { type: 'scan'; imageDataUrl: string; mode: ContentMode }

type GemmaState = {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null
  model: Awaited<ReturnType<typeof Gemma4ForConditionalGeneration.from_pretrained>> | null
  loadingPromise: Promise<void> | null
}

const state: GemmaState = {
  processor: null,
  model: null,
  loadingPromise: null,
}

function post(message: Record<string, unknown>) {
  self.postMessage(message)
}

function getPrompt(mode: ContentMode): string {
  if (mode === 'photo') {
    return [
      'Describe this image in detail.',
      'Include what you see: subjects, objects, setting, colors, composition, mood.',
      'Be thorough but concise. Use natural language.',
    ].join(' ')
  }

  if (mode === 'text') {
    return [
      'This image contains text. Perform OCR transcription.',
      'Return the full text exactly as written, preserving line breaks, formatting, and structure.',
      'Do not summarize, translate, or explain. Return only the transcribed text.',
      'After the full text, add a separator line "---" and then provide a brief 2-3 sentence summary of the content.',
    ].join(' ')
  }

  if (mode === 'mixed') {
    return [
      'This image contains both text and visual elements (like a textbook page, poster, or illustrated document).',
      'First, describe any images, diagrams, figures, or visual elements you see.',
      'Then, transcribe all text exactly as written, preserving formatting and structure.',
      'After the full transcription, add a separator "---" and provide a brief 2-3 sentence summary.',
      'Format your response as:',
      '## Visual Elements\n[description of images/diagrams]\n\n## Text Content\n[full transcription]\n\n---\n[summary]',
    ].join(' ')
  }

  // auto mode — detect and respond accordingly
  return [
    'Analyze this image and respond based on its content:',
    '- If it is primarily a photograph or illustration with no significant text, describe it in detail.',
    '- If it is primarily text (document, page, screenshot), transcribe all text exactly as written, preserving formatting. Then add "---" and a 2-3 sentence summary.',
    '- If it contains both text and visual elements (textbook, poster, infographic), first describe the visual elements, then transcribe all text. Then add "---" and a 2-3 sentence summary.',
    'Be thorough and accurate.',
  ].join(' ')
}

function sanitizeOutput(text: string) {
  return text
    .replace(/^```(?:text|markdown)?/i, '')
    .replace(/```$/i, '')
    .trim()
}

async function ensureLoaded() {
  // Use default HuggingFace CDN — no custom proxy needed
  env.allowLocalModels = false

  if (state.processor && state.model) {
    post({ status: 'ready' })
    return
  }

  if (state.loadingPromise) {
    await state.loadingPromise
    return
  }

  post({ status: 'loading', data: `Loading Gemma 4 model (${MODEL_ID})...` })

  const progress_callback = (info: any) => {
    if (info.status === 'progress_total') {
      post({ status: 'progress', progress: Math.round(info.progress ?? 0) })
      return
    }
    if (info.status === 'download') {
      post({ status: 'loading', data: `Downloading ${info.name ?? 'model shard'}...` })
    }
  }

  state.loadingPromise = Promise.all([
    AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: 'q4f16',
      device: 'webgpu',
      progress_callback,
    }),
  ])
    .then(([processor, model]) => {
      state.processor = processor
      state.model = model
      post({ status: 'ready' })
    })
    .catch((error) => {
      post({ status: 'error', data: error instanceof Error ? error.message : String(error) })
      throw error
    })
    .finally(() => {
      state.loadingPromise = null
    })

  await state.loadingPromise
}

async function runScan(imageDataUrl: string, mode: ContentMode) {
  await ensureLoaded()

  if (!state.processor || !state.model) {
    throw new Error('Gemma 4 model not available after load')
  }

  post({ status: 'scan-status', data: 'Preparing prompt', progress: 35 })

  const image = await load_image(imageDataUrl)
  const message = {
    role: 'user',
    content: [
      { type: 'image', image: imageDataUrl },
      { type: 'text', text: getPrompt(mode) },
    ],
  }

  const prompt = state.processor.apply_chat_template([message], {
    add_generation_prompt: true,
  })

  post({ status: 'scan-status', data: 'Encoding inputs', progress: 50 })

  const inputs = await state.processor(prompt, image, null, {
    add_special_tokens: false,
    max_soft_tokens: MAX_SOFT_TOKENS,
  } as any)

  post({ status: 'scan-status', data: 'Running Gemma 4 inference', progress: 65 })

  const outputs = await state.model.generate({
    ...inputs,
    max_new_tokens: 2048,
    do_sample: false,
  })

  post({ status: 'scan-status', data: 'Decoding output', progress: 92 })

  const promptLength = inputs.input_ids.dims.at(-1) ?? 0
  const generated = (outputs as any).slice(null, [promptLength, null])
  const decoded = state.processor.batch_decode(generated, {
    skip_special_tokens: true,
  })

  const text = sanitizeOutput(decoded[0] ?? '')

  post({
    status: 'scan-complete',
    data: { text, mode, modelId: MODEL_ID },
  })
}

self.addEventListener('message', async (event: MessageEvent<WorkerRequest>) => {
  try {
    if (event.data.type === 'load') {
      await ensureLoaded()
      return
    }
    if (event.data.type === 'scan') {
      await runScan(event.data.imageDataUrl, event.data.mode)
    }
  } catch (error) {
    post({ status: 'error', data: error instanceof Error ? error.message : String(error) })
  }
})
