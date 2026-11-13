import './app.css'
import { ApiError, ExpeditionApi, type SimulationResponse } from './api'
import {
  createContractMap,
  normalizeCatalog,
  normalizeEpisode,
  normalizeFrame,
  normalizeMap,
  normalizeReplayBundle,
} from './data'
import { ExpeditionScene } from './scene/expedition-scene'
import type { CatalogData, CatalogRun, EpisodeData, EpisodeSummary, ExpeditionMap, MonitorData, ReplayFrame, RunOption } from './types'
import { FRAME_INDEX } from './types'
import { LearningCurve } from './ui/curve'
import { MissionHud, required } from './ui/hud'
import { RegistryView } from './ui/registry'
import { TrainingDashboard } from './ui/training-dashboard'

type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const hud = new MissionHud()
const curve = new LearningCurve(required<HTMLCanvasElement>('#curve'))
const playButton = required<HTMLButtonElement>('#playButton')
const playIcon = required<HTMLElement>('#playIcon')
const timeline = required<HTMLInputElement>('#timeline')
const runSelect = required<HTMLSelectElement>('#runSelect')
const levelSelect = required<HTMLSelectElement>('#levelSelect')
const seedsInput = required<HTMLInputElement>('#seedsInput')
const evaluateButton = required<HTMLButtonElement>('#evaluateButton')
const manualButton = required<HTMLButtonElement>('#manualButton')
const manualEnvironment = required<HTMLSelectElement>('#manualEnvironment')
const difficultySelect = required<HTMLSelectElement>('#difficultySelect')
const difficultyTitle = required<HTMLElement>('#difficultyTitle')
const difficultyDescription = required<HTMLElement>('#difficultyDescription')
const difficultyFacts = required<HTMLElement>('#difficultyFacts')
const missionSeedValue = required<HTMLElement>('#missionSeedValue')
const planetLevelButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-planet-level]'))
const demoReplayButton = required<HTMLButtonElement>('#demoReplayButton')
const refreshEpisodesButton = required<HTMLButtonElement>('#refreshEpisodesButton')
const loadEpisodeButton = required<HTMLButtonElement>('#loadEpisodeButton')
const episodeNumberInput = required<HTMLInputElement>('#episodeNumberInput')
const episodeList = required<HTMLElement>('#episodeList')
const learningMeta = required<HTMLElement>('#learningMeta')
const keypadButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-action]'))
const trainedValue = required<HTMLElement>('#trainedValue')
const configValue = required<HTMLElement>('#configValue')

const scene = new ExpeditionScene(required<HTMLElement>('#scene'), {
  onFrame: (snapshot) => hud.updateFrame(snapshot),
  onPlaybackChange: (playing) => {
    playIcon.textContent = playing ? 'Ⅱ' : '▶'
    playButton.setAttribute('aria-label', playing ? 'Pausar' : 'Reproducir')
  },
})

let api: ExpeditionApi | undefined
let catalog: CatalogData = { runs: [] }
let currentMap = createContractMap()
let apiEnvironmentVersion = 'sin-versión'
let apiOnline = false
let manualSessionId: string | undefined
let manualEpisode: EpisodeData | undefined
let manualBusy = false
let lastVersion: Record<string, unknown> | undefined

const PLANETARY_LEVELS: Record<number, { title: string; description: string; facts: string[] }> = {
  "1": {
    "title": "01 · Primeros movimientos",
    "description": "Orientarse, recoger una muestra y regresar.",
    "facts": [
      "scenario_01",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "2": {
    "title": "02 · El primer desvío",
    "description": "Rodear una barrera; comparar dos rutas.",
    "facts": [
      "scenario_02",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "3": {
    "title": "03 · El laberinto rocoso",
    "description": "Intersecciones y pasillos con tres objetivos.",
    "facts": [
      "scenario_03",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "4": {
    "title": "04 · El cráter central",
    "description": "Rodear un cráter protegido por una formación rocosa.",
    "facts": [
      "scenario_04",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "5": {
    "title": "05 · El valle de decisiones",
    "description": "Elegir entre corredores y costes de tracción.",
    "facts": [
      "scenario_05",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "6": {
    "title": "06 · El laberinto de los cráteres",
    "description": "Ordenar cuatro muestras de distintos tipos.",
    "facts": [
      "scenario_06",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "7": {
    "title": "07 · La tormenta marciana",
    "description": "Resolver corredores con ruido, polvo y viento.",
    "facts": [
      "scenario_07",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "8": {
    "title": "08 · La ruta imposible a simple vista",
    "description": "Alejarse de objetivos para rodear barreras alternadas.",
    "facts": [
      "scenario_08",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "9": {
    "title": "09 · La expedición crítica",
    "description": "Seis objetivos, callejones y un presupuesto limitado.",
    "facts": [
      "scenario_09",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  },
  "10": {
    "title": "10 · La gran misión marciana",
    "description": "Ocho objetivos y más bifurcaciones: integrar navegación y recursos.",
    "facts": [
      "scenario_10",
      "mapa fijo",
      "semilla de sensores independiente"
    ]
  }
}

const registry = new RegistryView({
  onEvaluate: (run, levelId) => { void evaluateFromRegistry(run, levelId) },
  onRefreshMonitor: () => { void refreshMonitor() },
})
const trainingDashboard = new TrainingDashboard(required<HTMLCanvasElement>('#trainingDashboardCanvas'))

const setProvenance = (trained: string, config: string) => {
  trainedValue.textContent = trained
  configValue.textContent = config
}

scene.setMap(currentMap)
scene.setSpeed(4)
hud.reset()
setManualControls(false)

const errorMessage = (error: unknown) => {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return 'Ocurrió un error inesperado.'
}

const extractRecord = (value: unknown, key: string): UnknownRecord | undefined => {
  if (!isRecord(value)) return undefined
  return isRecord(value[key]) ? value[key] : undefined
}

const mapFromPayload = (value: unknown): ExpeditionMap | undefined => {
  if (!isRecord(value)) return undefined
  const candidate = value.map ?? extractRecord(value, 'episode')?.map ?? extractRecord(value, 'replay')?.map
  return candidate === undefined ? undefined : normalizeMap(candidate)
}

const frameFromResponse = (response: SimulationResponse): ReplayFrame | undefined => {
  const direct = normalizeFrame(response.frame)
  if (direct) return direct
  if (Array.isArray(response.frames)) return normalizeFrame(response.frames.at(-1))
  const state = isRecord(response.state) ? response.state : undefined
  return normalizeFrame(state?.frame)
}

const environmentVersionFrom = (value: unknown): string | undefined => {
  if (!isRecord(value)) return undefined
  for (const key of ['environment_version', 'served_version', 'version']) {
    if (typeof value[key] === 'string' && value[key]) return value[key]
  }
  return undefined
}

const parseSeeds = () => {
  const seeds = seedsInput.value
    .split(/[;,\s]+/)
    .filter(Boolean)
    .map(Number)
    .filter((value) => Number.isInteger(value))
  if (!seeds.length) throw new Error('Escribe al menos una semilla entera para evaluar.')
  return [...new Set(seeds)]
}

const updatePlanetaryLevel = () => {
  const level = Number(difficultySelect.value)
  const presentation = PLANETARY_LEVELS[level] ?? PLANETARY_LEVELS[1]
  difficultyTitle.textContent = presentation.title
  difficultyDescription.textContent = presentation.description
  difficultyFacts.replaceChildren(...presentation.facts.map((fact) => {
    const chip = document.createElement('span')
    chip.textContent = fact
    return chip
  }))
  for (const button of planetLevelButtons) {
    const selected = Number(button.dataset.planetLevel) === level
    button.classList.toggle('selected', selected)
    button.setAttribute('aria-pressed', String(selected))
  }
  if (!manualSessionId) manualButton.textContent = `Iniciar escenario L${level}`
}

const randomScenarioSeed = () => {
  const values = new Uint32Array(1)
  crypto.getRandomValues(values)
  return values[0] & 0x7fffffff
}

const setBusy = (button: HTMLButtonElement, busy: boolean, idleLabel: string, busyLabel: string) => {
  button.disabled = busy
  button.querySelector('span')?.replaceChildren(document.createTextNode(busy ? busyLabel : idleLabel))
}

const loadEpisode = (
  episode: EpisodeData,
  map: ExpeditionMap,
  source: 'api' | 'replay',
  sourceLabel?: string,
) => {
  currentMap = map
  scene.setMap(map)
  scene.setEpisode(episode)
  hud.setEpisode(episode, source, map)
  if (sourceLabel) hud.setSource(source, sourceLabel)
  curve.setData(episode.history, episode)
  timeline.max = String(Math.max(0, episode.frames.length - 1))
  timeline.value = '0'
  timeline.disabled = false
  playButton.disabled = false
}

const loadReplay = (value: unknown, label: string) => {
  const bundle = normalizeReplayBundle(value)
  loadEpisode(bundle.episodes[0], bundle.map, 'replay', `REPLAY · ${label}`)
  setProvenance('replay', label)
  scene.setPlaying(true)
  if (apiOnline) {
    hud.hideNotice()
  } else {
    hud.showNotice(
      'REPLAY activo · API no configurada',
      'El rover se mueve con evidencia guardada. El control manual seguirá deshabilitado hasta conectar la API HTTPS.',
    )
  }
}

const selectedRun = (): RunOption | undefined => catalog.runs.find((run) => run.runId === runSelect.value)

const fillCatalog = (data: CatalogData) => {
  catalog = data
  runSelect.replaceChildren()
  for (const run of data.runs) {
    const option = document.createElement('option')
    option.value = run.runId
    const metric = run.successRate === undefined ? '' : ` · ${(run.successRate * 100).toFixed(0)}% éxito`
    option.textContent = `${run.runId} · ${run.algorithm.toUpperCase()} · ${run.levelId.toUpperCase()}${metric}`
    runSelect.append(option)
  }
  if (!data.runs.length) {
    const option = document.createElement('option')
    option.value = ''
    option.textContent = 'Catálogo vacío'
    runSelect.append(option)
  }
  const recommended = data.recommendedByLevel?.[levelSelect.value] ?? data.recommendedRunId
  if (recommended && data.runs.some((run) => run.runId === recommended)) {
    runSelect.value = recommended
  }
  runSelect.disabled = !data.runs.length
  evaluateButton.disabled = !data.runs.length
  required<HTMLElement>('#catalogCount').textContent = `${data.runs.length} run${data.runs.length === 1 ? '' : 's'}`
  const run = selectedRun()
  if (run && hasLevel(run.levelId)) levelSelect.value = run.levelId
}

// Los baselines sirven en cualquier nivel (level_id "todos"): solo se copia un nivel que exista.
const hasLevel = (levelId: string) => [...levelSelect.options].some((option) => option.value === levelId)

const connectApi = async () => {
  try {
    api = new ExpeditionApi()
  } catch (error) {
    hud.setConnection('offline', 'Sin configurar')
    hud.showNotice(
      'API no configurada',
      `${errorMessage(error)} Define una API HTTPS o carga un archivo local, que aparecerá claramente como REPLAY.`,
    )
    return
  }

  hud.setConnection('loading', 'Conectando')
  try {
    await api.health()
    const [catalogResult, versionResult] = await Promise.allSettled([api.catalog(), api.version()])
    if (catalogResult.status === 'rejected') throw catalogResult.reason
    const normalizedCatalog = normalizeCatalog(catalogResult.value)
    fillCatalog(normalizedCatalog)
    registry.setCatalog(catalogResult.value)
    if (versionResult.status === 'fulfilled') {
      lastVersion = isRecord(versionResult.value) ? versionResult.value : undefined
      apiEnvironmentVersion = environmentVersionFrom(versionResult.value)
        ?? normalizedCatalog.servedVersion
        ?? apiEnvironmentVersion
    }
    hud.setConnection('online', 'API disponible')
    apiOnline = true
    manualButton.disabled = false
    refreshEpisodesButton.disabled = false
    loadEpisodeButton.disabled = false
    if (normalizedCatalog.runs.length) hud.hideNotice()
    else hud.showNotice('Catálogo vacío', 'La API responde, pero /catalog todavía no contiene modelos recuperables.')
    void refreshMonitor()
    void refreshEpisodes()
  } catch (error) {
    apiOnline = false
    hud.setConnection('offline', 'API sin respuesta')
    hud.showNotice('No fue posible conectar con la API', errorMessage(error))
  }
}

const loadStoredEpisode = async (episodeNumber: number) => {
  if (!api || !Number.isInteger(episodeNumber) || episodeNumber < 1) {
    hud.showNotice('Número de episodio inválido', 'Escribe un entero mayor que cero.')
    return
  }
  loadEpisodeButton.disabled = true
  try {
    const bundle = normalizeReplayBundle(await api.episodeReplay(episodeNumber))
    loadEpisode(bundle.episodes[0], bundle.map, 'replay', `REPLAY · EPISODIO ${episodeNumber}`)
    setProvenance('episodio persistido', `SQLite · #${episodeNumber}`)
    scene.setPlaying(true)
    hud.hideNotice()
  } catch (error) {
    hud.showNotice('No se pudo cargar el episodio', errorMessage(error))
  } finally {
    loadEpisodeButton.disabled = !apiOnline
  }
}

async function refreshEpisodes() {
  if (!api) return
  const activeApi = api
  refreshEpisodesButton.disabled = true
  try {
    const progress=await api.campaignProgress()
    if (isRecord(progress)) {
      const mastered=Array.isArray(progress.mastered) ? progress.mastered.length : 0
      required<HTMLElement>('#campaignProgress').textContent = `Currículo: escenario ${Number(progress.current_level ?? 1)} · ${mastered}/10 superados · ${10-mastered} pendientes${progress.run_id ? '' : ' · sin evaluación registrada'}`
    }
    const payload = await api.episodes(100, `scenario_${difficultySelect.value.padStart(2,'0')}`, required<HTMLSelectElement>('#episodeMode').value)
    const episodes = isRecord(payload) && Array.isArray(payload.episodes) ? payload.episodes.filter(isRecord) : []
    const storage = isRecord(payload) && isRecord(payload.storage) ? payload.storage : undefined
    required<HTMLElement>('#experimentStorageMeta').textContent = storage
      ? `${Number(storage.summarized_episodes ?? 0)} intentos registrados · historial completo, sin poda automática`
      : 'Se conserva cada intento completo, incluidos los fallidos.'
    const summaries: EpisodeSummary[] = episodes.map((item) => ({
      episode_number: Number(item.episode_number ?? 0),
      scenario_id: String(item.scenario_id ?? 'sin escenario'),
      difficulty_level: Number(item.difficulty_level ?? 1),
      requested_samples: Number(item.requested_samples ?? 0),
      samples_collected: Number(item.samples_collected ?? 0),
      success: typeof item.success === 'boolean' ? item.success : null,
      reward: Number(item.reward ?? 0),
      collisions: Number(item.collisions ?? 0),
      path_length: Number(item.path_length ?? 0),
      duration_s: Number(item.simulation_time_s ?? 0),
      energy_consumed: Number(item.energy_consumed ?? 0),
      path_efficiency: typeof item.path_efficiency === 'number' ? item.path_efficiency : null,
      end: String(item.end ?? 'en progreso'),
    })).filter((item) => Number.isInteger(item.episode_number) && item.episode_number > 0)
    const finished = summaries.filter((episode) => episode.success !== null)
    trainingDashboard.setEpisodes(finished)
    const successCount = finished.filter((episode) => episode.success).length
    learningMeta.textContent = summaries.length
      ? `${summaries.length} episodios recientes · ${finished.length ? ((successCount / finished.length) * 100).toFixed(1) : '0.0'}% de misiones completadas`
      : 'Sin episodios registrados.'
    episodeList.replaceChildren()
    if (!episodes.length) {
      const empty = document.createElement('p')
      empty.className = 'section-copy'
      empty.textContent = 'Aún no hay intentos planetarios guardados. Inicia un control manual para crear el primero.'
      episodeList.append(empty)
      return
    }
    for (const item of episodes.slice(0, 30)) {
      const number = Number(item.episode_number)
      if (!Number.isInteger(number)) continue
      const button = document.createElement('div')
      button.className = 'episode-item'
      button.tabIndex = 0
      button.setAttribute('role', 'button')
      const numberLabel = document.createElement('b')
      numberLabel.textContent = `#${number}`
      const description = document.createElement('span')
      description.textContent = `${String(item.scenario_id ?? 'sin escenario')} · ${Number(item.samples_collected ?? 0)}/${Number(item.requested_samples ?? 0)} muestras`
      const status = document.createElement('small')
      status.textContent = `${String(item.end ?? 'en progreso')} · L${Number(item.difficulty_level ?? 1)} · ${String(item.terrain_signature ?? 'terreno legado')}`
      const protect = document.createElement('button')
      protect.type = 'button'
      protect.className = `episode-protect${item.protected ? ' active' : ''}`
      protect.title = item.protected ? 'Quitar protección manual' : 'Conservar este experimento'
      protect.setAttribute('aria-label', protect.title)
      protect.textContent = item.protected ? '★' : '☆'
      protect.addEventListener('click', async (event) => {
        event.stopPropagation()
        protect.disabled = true
        try {
          await activeApi.protectEpisode(number, !Boolean(item.protected))
          await refreshEpisodes()
        } catch (error) {
          hud.showNotice('No se pudo cambiar la protección', errorMessage(error))
          protect.disabled = false
        }
      })
      button.append(numberLabel, description, status, protect)
      button.addEventListener('click', () => {
        episodeNumberInput.value = String(number)
        void loadStoredEpisode(number)
      })
      button.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          episodeNumberInput.value = String(number)
          void loadStoredEpisode(number)
        }
      })
      episodeList.append(button)
    }
  } catch (error) {
    episodeList.replaceChildren()
    const message = document.createElement('p')
    message.className = 'section-copy'
    message.textContent = errorMessage(error)
    episodeList.append(message)
  } finally {
    refreshEpisodesButton.disabled = !apiOnline
  }
}

const evaluateSelectedRun = async () => {
  if (!api) return
  const run = selectedRun()
  if (!run) {
    hud.showNotice('Selecciona un modelo', 'El catálogo no contiene un run válido para evaluar.')
    return
  }

  const rich = registry.run(run.runId)
  if (rich && !registry.canServe(rich)) {
    hud.showNotice('Modelo no evaluable',
      `El run ${run.runId} tiene estado ${rich.status} o un ambiente incompatible (${rich.environment_version}). Se conserva como evidencia, pero no se sirve.`)
    return
  }

  setBusy(evaluateButton, true, 'Evaluar modelo', 'Evaluando…')
  try {
    const result = await api.evaluate(run.runId, levelSelect.value, parseSeeds())
    const episode = normalizeEpisode(result, {
      mode: 'evaluación',
      runId: run.runId,
      modelName: run.modelName,
      environmentVersion: run.environmentVersion || apiEnvironmentVersion,
      levelId: levelSelect.value,
    })
    loadEpisode(episode, mapFromPayload(result) ?? currentMap, 'api')
    const trained = rich?.level_id ?? run.levelId
    const crossLevel = trained !== 'todos' && trained !== levelSelect.value
    setProvenance(
      crossLevel ? `${trained.toUpperCase()} · evaluado en ${levelSelect.value.toUpperCase()}` : trained.toUpperCase(),
      rich?.config_path ? `${rich.experiment_id} · semilla ${rich.seed ?? '—'}` : rich?.experiment_id ?? '—',
    )
    if (crossLevel) {
      hud.showNotice('Evaluación en otro nivel',
        `Este modelo se entrenó en ${trained.toUpperCase()} y se está evaluando en ${levelSelect.value.toUpperCase()}. El contrato de observación y acciones es el mismo; el resultado mide transferencia.`)
    } else {
      hud.hideNotice()
    }
  } catch (error) {
    hud.showNotice('La evaluación falló', errorMessage(error))
  } finally {
    setBusy(evaluateButton, false, 'Evaluar modelo', 'Evaluando…')
    evaluateButton.disabled = !catalog.runs.length
    void refreshMonitor()
  }
}

async function evaluateFromRegistry(run: CatalogRun, levelId: string) {
  if (!api) {
    hud.showNotice('API no configurada', 'Conecta la API HTTPS para cargar y evaluar modelos.')
    return
  }
  if (![...runSelect.options].some((option) => option.value === run.run_id)) {
    hud.showNotice('Run no disponible', `El run ${run.run_id} no está en el catálogo servido por la API.`)
    return
  }
  runSelect.value = run.run_id
  if (hasLevel(levelId)) levelSelect.value = levelId
  document.querySelector('#mission')?.scrollIntoView({ behavior: 'smooth' })
  await evaluateSelectedRun()
}

async function refreshMonitor() {
  if (!api) return
  try {
    const [monitor, version] = await Promise.all([api.monitor(), api.version()])
    lastVersion = isRecord(version) ? version : lastVersion
    registry.renderMonitor(monitor as MonitorData, lastVersion)
  } catch (error) {
    registry.renderMonitor(undefined, lastVersion, errorMessage(error))
  }
}

const startManualMode = async () => {
  if (!api) return
  manualButton.disabled = true
  manualButton.textContent = 'Iniciando…'
  try {
    const environment = manualEnvironment.value === 'planetary' ? 'planetary' : 'beacon'
    const difficulty = Number(difficultySelect.value)
    const seed = environment === 'planetary' ? randomScenarioSeed() : 0
    const response = await api.reset(levelSelect.value, environment, difficulty, seed, manualSessionId)
    const frame = frameFromResponse(response)
    if (!response.session_id) throw new Error('/sim/reset no devolvió session_id.')
    if (!frame) throw new Error('/sim/reset no devolvió un cuadro compatible.')
    manualSessionId = response.session_id
    const map = response.map ? normalizeMap(response.map) : currentMap
    missionSeedValue.textContent = environment === 'planetary'
      ? `semilla ${String(response.seed ?? seed)} · ${map.scenario_id ?? 'escenario fijo'}`
      : 'modo de balizas'
    const initial = normalizeEpisode(
      { ...response, frames: [frame], episode: 'manual', end: 'running' },
      {
        mode: 'manual',
        runId: 'control-manual',
        modelName: 'sin-modelo',
        environmentVersion: environmentVersionFrom(response) ?? apiEnvironmentVersion,
        levelId: environment === 'planetary' ? `L${difficulty}` : levelSelect.value,
      },
    )
    manualEpisode = environment === 'planetary' ? {
      ...initial,
      scenarioId: map.scenario_id,
      difficulty: map.difficulty,
      planet: map.environment?.planet?.planet_id ?? 'mars',
      objective: `Recolectar ${map.mission?.requested_samples ?? 0} × ${map.mission?.target_type ?? 'muestra'}`,
      requestedSamples: map.mission?.requested_samples,
      samplesCollected: 0,
      missionTimeline: [{
        target_type: map.mission?.target_type,
        requested: map.mission?.requested_samples,
        collected: 0,
        inventory: [],
        samples: map.samples,
        status: 'recolectando',
      }],
    } : initial
    loadEpisode(manualEpisode, map, 'api', 'API · MANUAL')
    setProvenance('sin modelo', environment === 'planetary' ? `escenario fijo · L${difficulty}` : 'control manual heredado')
    setManualControls(true)
    hud.hideNotice()
  } catch (error) {
    manualSessionId = undefined
    manualEpisode = undefined
    setManualControls(false)
    hud.showNotice('No se pudo iniciar el control manual', errorMessage(error))
  } finally {
    manualButton.disabled = false
    manualButton.textContent = manualEnvironment.value === 'planetary'
      ? `${manualSessionId ? 'Nueva' : 'Generar'} misión L${difficultySelect.value}`
      : `${manualSessionId ? 'Reiniciar' : 'Iniciar'} balizas`
  }
}

async function sendManualAction(action: number) {
  if (!api || !manualSessionId || !manualEpisode || manualBusy) return
  manualBusy = true
  setManualControls(false)
  try {
    const response = await api.step(manualSessionId, action)
    const frame = frameFromResponse(response)
    if (!frame) throw new Error('/sim/step no devolvió un cuadro compatible.')
    const frames = [...manualEpisode.frames, frame]
    const finished = Boolean(response.terminated || response.truncated)
    const map = response.map ? normalizeMap(response.map) : currentMap
    // La etapa 3 es siempre "volver a la base": en N1 se pasa de 1 a 3, con una sola baliza.
    const visitedNow = Math.min(frame[FRAME_INDEX.stage] - 1, map.beacons.length)
    manualEpisode = {
      ...manualEpisode,
      frames,
      steps: frames.length - 1,
      battery: frame[FRAME_INDEX.battery],
      reward: frames.reduce((total, item) => total + item[FRAME_INDEX.reward], 0),
      visited: Math.max(manualEpisode.visited, visitedNow),
      success: response.end === 'exito' || manualEpisode.success,
      end: typeof response.end === 'string' ? response.end : finished ? 'finished' : 'running',
      samplesCollected: typeof response.samples_collected === 'number'
        ? response.samples_collected
        : manualEpisode.samplesCollected,
      requestedSamples: typeof response.requested_samples === 'number'
        ? response.requested_samples
        : manualEpisode.requestedSamples,
      missionTimeline: manualEpisode.requestedSamples === undefined ? manualEpisode.missionTimeline : [
        ...(manualEpisode.missionTimeline ?? []),
        {
          target_type: typeof response.target_type === 'string' ? response.target_type : map.mission?.target_type,
          requested: typeof response.requested_samples === 'number' ? response.requested_samples : map.mission?.requested_samples,
          collected: typeof response.samples_collected === 'number' ? response.samples_collected : 0,
          inventory: Array.isArray(response.inventory) ? response.inventory.filter((item): item is string => typeof item === 'string') : [],
          samples: (map.samples ?? []).map((sample) => ({
            ...sample,
            collected: Array.isArray(response.inventory) && response.inventory.includes(sample.sample_id),
          })),
          event: typeof response.event === 'string' ? response.event : null,
          status: finished ? 'finalizada' : 'recolectando',
        },
      ],
    }
    loadEpisode(manualEpisode, map, 'api', 'API · MANUAL')
    scene.seek(frames.length - 1)
    if (finished) {
      manualSessionId = undefined
      hud.showNotice('Episodio finalizado', `Causa de fin: ${manualEpisode.end}. Pulsa Reiniciar para conducir de nuevo.`)
      void refreshEpisodes()
    }
  } catch (error) {
    hud.showNotice('Falló la orden manual', errorMessage(error))
  } finally {
    manualBusy = false
    setManualControls(Boolean(manualSessionId))
  }
}

function setManualControls(enabled: boolean) {
  keypadButtons.forEach((button) => {
    button.disabled = !enabled || (Number(button.dataset.action) >= 5 && manualEnvironment.value !== 'planetary')
  })
}

playButton.addEventListener('click', () => scene.togglePlayback())
timeline.addEventListener('input', () => {
  scene.setPlaying(false)
  scene.seek(Number(timeline.value))
})
required<HTMLSelectElement>('#speedSelect').addEventListener('change', (event) => {
  scene.setSpeed(Number((event.currentTarget as HTMLSelectElement).value))
})

required<HTMLButtonElement>('#followButton').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement
  const active = button.getAttribute('aria-pressed') !== 'true'
  button.setAttribute('aria-pressed', String(active))
  scene.setFollow(active)
})

required<HTMLButtonElement>('#lightModeButton').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement
  const active = button.getAttribute('aria-pressed') !== 'true'
  button.setAttribute('aria-pressed', String(active))
  scene.setLightMode(active)
})

runSelect.addEventListener('change', () => {
  const run = selectedRun()
  if (run && hasLevel(run.levelId)) levelSelect.value = run.levelId
})
levelSelect.addEventListener('change', () => {
  const recommended = catalog.recommendedByLevel?.[levelSelect.value]
  if (recommended && [...runSelect.options].some((option) => option.value === recommended)) runSelect.value = recommended
})
evaluateButton.addEventListener('click', () => { void evaluateSelectedRun() })
manualButton.addEventListener('click', () => { void startManualMode() })
for (const button of planetLevelButtons) {
  button.addEventListener('click', () => {
    difficultySelect.value = button.dataset.planetLevel ?? '1'
    manualEnvironment.value = 'planetary'
    updatePlanetaryLevel()
    void refreshEpisodes()
    document.querySelector('#planetaryMission')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}
difficultySelect.addEventListener('change', () => { updatePlanetaryLevel(); void refreshEpisodes() })
manualEnvironment.addEventListener('change', () => {
  difficultySelect.disabled = manualEnvironment.value !== 'planetary'
  setManualControls(Boolean(manualSessionId))
  manualButton.textContent = manualEnvironment.value === 'planetary'
    ? `Iniciar escenario L${difficultySelect.value}`
    : 'Iniciar balizas'
})
refreshEpisodesButton.addEventListener('click', () => { void refreshEpisodes() })
loadEpisodeButton.addEventListener('click', () => { void loadStoredEpisode(Number(episodeNumberInput.value)) })
episodeNumberInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') void loadStoredEpisode(Number(episodeNumberInput.value))
})

keypadButtons.forEach((button) => {
  button.addEventListener('click', () => { void sendManualAction(Number(button.dataset.action)) })
})

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return
  const actions: Partial<Record<string, number>> = { ArrowUp: 0, ArrowLeft: 3, ArrowRight: 4, r: 5, R: 5 }
  const action = actions[event.key]
  if (action === undefined || !manualSessionId) return
  event.preventDefault()
  void sendManualAction(action)
})

demoReplayButton.addEventListener('click', async () => {
  setBusy(demoReplayButton, true, 'Ver demo REPLAY', 'Cargando…')
  try {
    const response = await fetch('/demo/expedition_replays.json', { cache: 'force-cache' })
    if (!response.ok) throw new Error(`No fue posible cargar el replay incluido (${response.status}).`)
    loadReplay(await response.json() as unknown, 'demo incluido')
  } catch (error) {
    hud.showNotice('No fue posible cargar la demo', errorMessage(error))
  } finally {
    setBusy(demoReplayButton, false, 'Ver demo REPLAY', 'Cargando…')
  }
})

required<HTMLInputElement>('#replayInput').addEventListener('change', async (event) => {
  const input = event.currentTarget as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  try {
    loadReplay(JSON.parse(await file.text()) as unknown, file.name)
  } catch (error) {
    hud.showNotice('Replay incompatible', errorMessage(error))
  } finally {
    input.value = ''
  }
})

void connectApi()
updatePlanetaryLevel()

required<HTMLInputElement>('#showTrail').addEventListener('change',event => scene.setTrailVisible((event.target as HTMLInputElement).checked))
required<HTMLInputElement>('#showReference').addEventListener('change',event => scene.setReferenceVisible((event.target as HTMLInputElement).checked))
required<HTMLSelectElement>('#episodeMode').addEventListener('change',() => { void refreshEpisodes() })
required<HTMLButtonElement>('#compareAttempts').addEventListener('click',async () => {
  if (!api) return
  try {
    const ids=required<HTMLInputElement>('#compareNumbers').value.split(/[,;\s]+/).filter(Boolean).map(Number)
    const result=await api.compareEpisodes(ids)
    renderComparison(result)
  } catch(error) { required<HTMLElement>('#comparisonResult').textContent=errorMessage(error) }
})
required<HTMLButtonElement>('#scenarioProgress').addEventListener('click',async () => {
  if (!api) return
  try {
    const result=await api.scenarioProgress(`scenario_${difficultySelect.value.padStart(2,'0')}`,required<HTMLSelectElement>('#episodeMode').value)
    renderComparison(result)
  } catch(error) { required<HTMLElement>('#comparisonResult').textContent=errorMessage(error) }
})

function renderComparison(payload: unknown) {
  const output=required<HTMLElement>('#comparisonResult')
  output.replaceChildren()
  if (!isRecord(payload)) return
  const heading=document.createElement('h3')
  heading.textContent=payload.scenario_id ? `Aprendizaje · ${String(payload.scenario_id)}` : 'Comparación de intentos'
  output.append(heading)
  if (typeof payload.episodes==='number') {
    const summary=document.createElement('p')
    summary.textContent=`${payload.episodes} intentos · éxito ${(Number(payload.success_rate ?? 0)*100).toFixed(1)}% · recompensa media ${Number(payload.reward_mean ?? 0).toFixed(2)}`
    output.append(summary)
  }
  const groups: [string, unknown][] = Array.isArray(payload.episodes)
    ? [['Intentos seleccionados',payload.episodes]]
    : [['Primeros',payload.first],['Intermedios',payload.middle],['Recientes',payload.recent],['Mejor ruta completada',payload.best ? [payload.best] : []]]
  for (const [label,items] of groups) {
    if (!Array.isArray(items)) continue
    const title=document.createElement('h4');title.textContent=label;output.append(title)
    const table=document.createElement('table');table.style.width='100%'
    const header=document.createElement('tr')
    for (const name of ['Intento','Resultado','Recompensa','Choques','Distancia','Energía','Eficiencia']) { const cell=document.createElement('th');cell.textContent=name;header.append(cell) }
    table.append(header)
    for (const item of items.filter(isRecord)) {
      const row=document.createElement('tr')
      const values=[`#${item.episode_number}`,String(item.end),Number(item.reward).toFixed(1),String(item.collisions),`${Number(item.path_length).toFixed(1)} m`,Number(item.energy_consumed ?? 0).toFixed(1),typeof item.path_efficiency==='number' ? item.path_efficiency.toFixed(3) : '—']
      for (const value of values) {const cell=document.createElement('td');cell.textContent=value;row.append(cell)}
      row.style.cursor='pointer';row.title='Reproducir este intento'
      row.addEventListener('click',()=>{ void loadStoredEpisode(Number(item.episode_number)) })
      table.append(row)
    }
    output.append(table)
  }
  const policies=Array.isArray(payload.evaluated_policies) ? payload.evaluated_policies.filter(isRecord) : []
  const status=document.createElement('p')
  status.textContent=policies.length ? `Mejor política evaluada: ${String(policies[0].policy_version)} · ${Number(policies[0].episodes)} pruebas · éxito ${(Number(policies[0].success_rate)*100).toFixed(1)}%` : 'Sin evaluación de políticas suficiente para afirmar aprendizaje.'
  output.append(status)
}
