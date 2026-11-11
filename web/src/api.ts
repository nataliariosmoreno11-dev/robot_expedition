import type { CatalogData, ExpeditionMap, ReplayFrame } from './types'

const DEFAULT_TIMEOUT_MS = 12_000

export class ApiError extends Error {
  status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export interface SimulationResponse {
  session_id?: string
  frame?: ReplayFrame
  frames?: ReplayFrame[]
  map?: ExpeditionMap
  terminated?: boolean
  truncated?: boolean
  end?: string
  [key: string]: unknown
}

export class ExpeditionApi {
  readonly baseUrl: string

  constructor(rawBaseUrl: string | undefined = import.meta.env.VITE_API_BASE_URL) {
    if (!rawBaseUrl?.trim()) {
      throw new ApiError('VITE_API_BASE_URL no está configurada.')
    }

    let parsed: URL
    try {
      parsed = new URL(rawBaseUrl.trim())
    } catch {
      throw new ApiError('VITE_API_BASE_URL debe ser una URL absoluta con HTTPS.')
    }

    const localHosts = new Set(['localhost', '127.0.0.1', '::1'])
    if (import.meta.env.PROD && localHosts.has(parsed.hostname)) {
      throw new ApiError('La compilación de producción no puede apuntar a localhost.')
    }
    if (import.meta.env.PROD && parsed.protocol !== 'https:') {
      throw new ApiError('La API de producción debe usar HTTPS.')
    }

    this.baseUrl = parsed.toString().replace(/\/$/, '')
  }

  async health(timeoutMs = DEFAULT_TIMEOUT_MS): Promise<unknown> {
    return this.request('/health', {}, timeoutMs)
  }

  async catalog(): Promise<CatalogData | unknown> {
    return this.request('/catalog')
  }

  async version(): Promise<unknown> {
    return this.request('/version')
  }

  async monitor(): Promise<unknown> {
    return this.request('/monitor')
  }

  async evaluate(runId: string, levelId: string, seeds: number[]): Promise<unknown> {
    return this.request('/evaluate', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId, level_id: levelId, seeds }),
    })
  }

  async reset(levelId: string): Promise<SimulationResponse> {
    return this.request('/sim/reset', {
      method: 'POST',
      body: JSON.stringify({ level_id: levelId }),
    }) as Promise<SimulationResponse>
  }

  async step(sessionId: string, action: number): Promise<SimulationResponse> {
    return this.request('/sim/step', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, action }),
    }) as Promise<SimulationResponse>
  }

  private async request(
    path: string,
    init: RequestInit = {},
    timeoutMs = DEFAULT_TIMEOUT_MS,
  ): Promise<unknown> {
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs)

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
        cache: 'no-store',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
          ...init.headers,
        },
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new ApiError(`${response.status} ${response.statusText}${detail ? ` · ${detail.slice(0, 180)}` : ''}`, response.status)
      }

      const contentType = response.headers.get('content-type') ?? ''
      if (!contentType.includes('application/json')) {
        throw new ApiError(`La respuesta de ${path} no es JSON.`)
      }
      return await response.json()
    } catch (error) {
      if (error instanceof ApiError) throw error
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new ApiError(`La solicitud a ${path} superó ${Math.round(timeoutMs / 1000)} segundos.`)
      }
      throw new ApiError(error instanceof Error ? error.message : `Error de red en ${path}.`)
    } finally {
      window.clearTimeout(timeout)
    }
  }
}
