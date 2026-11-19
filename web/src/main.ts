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
import { mountAgentTraining } from './ui/agent-training'
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
const qTrainingStatus = required<HTMLElement>('#qTrainingStatus')
const episodeModelCheckpoint = required<HTMLElement>('#episodeModelCheckpoint')
const episodeModelStates = required<HTMLElement>('#episodeModelStates')
const episodeModelConnections = required<HTMLElement>('#episodeModelConnections')
const episodeModelUpdates = required<HTMLElement>('#episodeModelUpdates')
const episodeModelEpsilon = required<HTMLElement>('#episodeModelEpsilon')
const episodeModelRates = required<HTMLElement>('#episodeModelRates')
const episodeModelDecisions = required<HTMLElement>('#episodeModelDecisions')
const checkpointSelect = required<HTMLSelectElement>('#checkpointSelect')
const checkpointMeta = required<HTMLElement>('#checkpointMeta')
const activateCheckpointButton = required<HTMLButtonElement>('#activateCheckpointButton')
const snapshotCheckpointButton = required<HTMLButtonElement>('#snapshotCheckpointButton')
const refreshCheckpointsButton = required<HTMLButtonElement>('#refreshCheckpointsButton')
const keypadButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-action]'))
const trainedValue = required<HTMLElement>('#trainedValue')
const configValue = required<HTMLElement>('#configValue')
const missionMinimap = required<HTMLCanvasElement>('#missionMinimap')
const elevationMap = required<HTMLCanvasElement>('#elevationMap')
const ppoCampaignMeta = required<HTMLElement>('#ppoCampaignMeta')
const ppoModelSwitch = required<HTMLElement>('#ppoModelSwitch')
const ppoCampaignLevels = required<HTMLElement>('#ppoCampaignLevels')
const ppoEvidenceDetail = required<HTMLElement>('#ppoEvidenceDetail')
const ppoStart = required<HTMLButtonElement>('#ppoStart')
const ppoPause = required<HTMLButtonElement>('#ppoPause')
const ppoRestart = required<HTMLButtonElement>('#ppoRestart')
const ppoFullscreen = required<HTMLButtonElement>('#ppoFullscreen')
const ppoDecisionHistory = required<HTMLElement>('#ppoDecisionHistory')
const ppoActionProbabilities = required<HTMLElement>('#ppoActionProbabilities')

const scene = new ExpeditionScene(required<HTMLElement>('#scene'), {
  onFrame: (snapshot) => {
    hud.updateFrame(snapshot)
    updateReferenceOverlay(snapshot.frame)
  },
  onPlaybackChange: (playing) => {
    playIcon.textContent = playing ? 'Ⅱ' : '▶'
    playButton.setAttribute('aria-label', playing ? 'Pausar' : 'Reproducir')
  },
})
const ppoScene = new ExpeditionScene(required<HTMLElement>('#ppoScene'), {
  onFrame: () => undefined,
  onPlaybackChange: () => undefined,
})

let api: ExpeditionApi | undefined
let catalog: CatalogData = { runs: [] }
let currentMap = createContractMap()
let apiEnvironmentVersion = 'sin-versión'
let apiOnline = false
let manualSessionId: string | undefined
let manualEpisode: EpisodeData | undefined
let manualBusy = false
let automaticRunning = false
let automaticTimer: number | undefined
let automaticEpisodeCount = 0
let automaticNetworkRetries = 0
let lastVersion: Record<string, unknown> | undefined
let latestQTraining: UnknownRecord | undefined
let displayedEpisode: EpisodeData | undefined
let selectedPpoLevel = 1
let selectedPpoRun = ''
let ppoSessionId: string | undefined
let ppoEpisode: EpisodeData | undefined
let ppoMap: ExpeditionMap | undefined
let ppoRunning = false
let ppoTimer: number | undefined
let ppoDecisionRows: UnknownRecord[] = []
let ppoCampaignModels: Array<{ level: number; runId: string }> = []
let ppoTriedRuns = new Set<string>()

function mapPoint(map: ExpeditionMap, canvas: HTMLCanvasElement, x: number, y: number): [number, number] {
  const width = map.active_extent?.[0] ?? map.size[0]
  const height = map.active_extent?.[1] ?? map.size[1]
  return [8 + (x / width) * (canvas.width - 16), 8 + ((height / 2 - y) / height) * (canvas.height - 16)]
}

function drawElevation(map: ExpeditionMap) {
  const context = elevationMap.getContext('2d')
  if (!context) return
  const n = Math.max(1, map.terrain.n)
  const values = map.terrain.heights
  const maximum = Math.max(...values, 0.001)
  const minimum = Math.min(...values, 0)
  const range = Math.max(0.001, maximum - minimum)
  context.clearRect(0, 0, elevationMap.width, elevationMap.height)
  for (let row = 0; row < n; row += 1) {
    for (let column = 0; column < n; column += 1) {
      const normalized = ((values[row * n + column] ?? 0) - minimum) / range
      const hue = 220 - normalized * 200
      context.fillStyle = `hsl(${hue} 78% ${32 + normalized * 28}%)`
      context.fillRect(column * elevationMap.width / n, row * elevationMap.height / n,
        Math.ceil(elevationMap.width / n), Math.ceil(elevationMap.height / n))
    }
  }
}

function drawMinimap(map: ExpeditionMap, frame?: ReplayFrame) {
  const context = missionMinimap.getContext('2d')
  if (!context) return
  context.clearRect(0, 0, missionMinimap.width, missionMinimap.height)
  context.fillStyle = '#552316'
  context.fillRect(0, 0, missionMinimap.width, missionMinimap.height)
  context.strokeStyle = 'rgba(255,215,170,.12)'
  context.lineWidth = 1
  for (let index = 1; index < 6; index += 1) {
    context.beginPath(); context.moveTo(index * missionMinimap.width / 6, 0); context.lineTo(index * missionMinimap.width / 6, missionMinimap.height); context.stroke()
    context.beginPath(); context.moveTo(0, index * missionMinimap.height / 6); context.lineTo(missionMinimap.width, index * missionMinimap.height / 6); context.stroke()
  }
  context.fillStyle = '#80614b'
  for (const rock of map.rocks) {
    const [x, y] = mapPoint(map, missionMinimap, rock[1], rock[2])
    context.beginPath(); context.arc(x, y, 2.5, 0, Math.PI * 2); context.fill()
  }
  for (const sample of map.samples ?? []) {
    const [x, y] = mapPoint(map, missionMinimap, sample.x, sample.y)
    context.fillStyle = sample.target ? '#59d9ef' : '#ffc058'
    context.beginPath(); context.arc(x, y, sample.collected ? 2 : 4, 0, Math.PI * 2); context.fill()
  }
  const [baseX, baseY] = mapPoint(map, missionMinimap, map.base[0], map.base[1])
  context.fillStyle = '#8adb6f'; context.fillRect(baseX - 4, baseY - 4, 8, 8)
  const route = displayedEpisode?.frames ?? []
  if (route.length > 1) {
    context.save()
    context.strokeStyle = '#61d5d2'
    context.lineWidth = 2.5
    context.shadowColor = '#61d5d2'
    context.shadowBlur = 5
    context.beginPath()
    route.forEach((routeFrame, index) => {
      const [routeX, routeY] = mapPoint(map, missionMinimap, routeFrame[0], routeFrame[1])
      if (index === 0) context.moveTo(routeX, routeY)
      else context.lineTo(routeX, routeY)
    })
    context.stroke()
    context.restore()
  }
  if (frame) {
    const [x, y] = mapPoint(map, missionMinimap, frame[0], frame[1])
    context.fillStyle = '#fff'; context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fill()
  }
}

function updateReferenceOverlay(frame: ReplayFrame) {
  required<HTMLElement>('#autoEnergy').textContent = `${Math.max(0, frame[FRAME_INDEX.battery]).toFixed(0)}%`
  required<HTMLElement>('#autoCargo').textContent = `${manualEpisode?.samplesCollected ?? 0}/${manualEpisode?.requestedSamples ?? currentMap.mission?.requested_samples ?? 0}`
  const collisions = manualEpisode?.frames.reduce((total, item) => total + item[FRAME_INDEX.collision], 0) ?? 0
  required<HTMLElement>('#autoHealth').textContent = `${Math.max(0, 100 - collisions * 8)}%`
  drawMinimap(currentMap, frame)
}

function updateReferenceMap(map: ExpeditionMap) {
  drawElevation(map)
  drawMinimap(map)
  required<HTMLElement>('#obstacleCount').textContent = `${map.rocks.length} rocas`
  required<HTMLElement>('#terrainKind').textContent = Number(map.difficulty_level ?? 1) >= 5 ? 'cráteres y dunas' : 'rocoso marciano'
  const dust = Number(map.environment?.dust_opacity ?? 0)
  required<HTMLElement>('#weatherCondition').textContent = dust > 0.45 ? 'tormenta de polvo' : dust > 0.2 ? 'polvo variable' : 'estable'
}

const PLANETARY_LEVELS: Record<number, { title: string; description: string; facts: string[] }> = {
  1: {
    title: 'L1 · Llanura inicial',
    description: 'Mapa pequeño y casi plano para aprender navegación, recolección y regreso a base.',
    facts: ['área 4.5 × 4.4 m', '1 muestra', '1 roca', 'plano y despejado'],
  },
  2: {
    title: 'L2 · Rocas dispersas',
    description: 'El área crece y aparecen más rutas, rocas, una duna y muestras distractoras.',
    facts: ['área 5.8 × 5.6 m', '1–2 muestras', '3 rocas', '1 duna'],
  },
  3: {
    title: 'L3 · Pendientes y fricción',
    description: 'Dos muestras objetivo, relieve más marcado y zonas donde las ruedas pierden tracción.',
    facts: ['área 7 × 7 m', '2 muestras', '5 rocas', '2 dunas'],
  },
  4: {
    title: 'L4 · Cráteres y viento',
    description: 'Rutas más largas con cráteres, viento variable y hasta tres muestras requeridas.',
    facts: ['área 8.2 × 8.2 m', '7 rocas', '2 cráteres', 'viento ≤ 11 m/s'],
  },
  5: {
    title: 'L5 · Dunas y polvo',
    description: 'Más dunas y distractores; el polvo reduce la distancia a la que el rover detecta muestras.',
    facts: ['área 9.2 × 9.2 m', '9 rocas', '5 dunas', '3 cráteres'],
  },
  6: {
    title: 'L6 · Rutas extensas',
    description: 'Mapa amplio, tres o cuatro muestras, varios cráteres y presupuesto de batería menor.',
    facts: ['área 10.2 × 10.2 m', '12 rocas', '7 dunas', '5 cráteres'],
  },
  7: {
    title: 'L7 · Meteorología variable',
    description: 'Polvo intenso, viento fuerte, regiones peligrosas y poca tracción exigen rutas seguras.',
    facts: ['área 11.1 × 10.7 m', '15 rocas', '7 cráteres', 'tormentas fuertes'],
  },
  8: {
    title: 'L8 · Expedición marciana',
    description: 'La misión completa combina el mapa mayor, cuatro muestras, dunas, cráteres, polvo y energía limitada.',
    facts: ['área 11.8 × 11.5 m', '18 rocas', '10 cráteres', '12 dunas · polvo máximo'],
  },
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

const showQTraining = (value: unknown) => {
  if (!isRecord(value)) return
  latestQTraining = value
  const episode = Number(value.episodes ?? 0)
  const states = Number(value.states ?? 0)
  const updates = Number(value.updates ?? 0)
  const epsilon = Number(value.epsilon ?? 0)
  const guided = Number(value.guided_decisions ?? 0)
  const random = Number(value.random_decisions ?? 0)
  const escapes = Number(value.loop_escapes ?? 0)
  const totalDecisions = guided + random + Number(value.greedy_decisions ?? 0)
  const guidance = totalDecisions ? ` · guía ${((guided / totalDecisions) * 100).toFixed(0)}%` : ''
  qTrainingStatus.textContent = `Q-learning · episodio ${episode} · ${states} estados · ${updates} actualizaciones · ε ${epsilon.toFixed(3)}${guidance}${escapes ? ` · ${escapes} escapes de ciclo` : ''}`
  episodeModelCheckpoint.textContent = String(value.checkpoint ?? value.active_checkpoint ?? 'checkpoint activo')
  episodeModelCheckpoint.title = episodeModelCheckpoint.textContent
  episodeModelStates.textContent = states.toLocaleString('es-CO')
  episodeModelConnections.textContent = `${(states * 7).toLocaleString('es-CO')} conexiones Q`
  episodeModelUpdates.textContent = updates.toLocaleString('es-CO')
  episodeModelEpsilon.textContent = epsilon.toFixed(3)
  episodeModelRates.textContent = `${Number(value.alpha ?? 0).toFixed(2)} / ${Number(value.gamma ?? 0).toFixed(2)}`
  episodeModelDecisions.textContent = totalDecisions.toLocaleString('es-CO')
}

async function refreshCheckpoints() {
  if (!api) return
  refreshCheckpointsButton.disabled = true
  try {
    const payload = await api.qlearningCheckpoints()
    const checkpoints = isRecord(payload) && Array.isArray(payload.checkpoints)
      ? payload.checkpoints.filter(isRecord)
      : []
    checkpointSelect.replaceChildren()
    for (const checkpoint of checkpoints) {
      const option = document.createElement('option')
      option.value = String(checkpoint.name ?? '')
      option.textContent = `${checkpoint.active ? '● ' : ''}${String(checkpoint.name ?? 'checkpoint')} · ep ${Number(checkpoint.episodes ?? 0)} · ${Number(checkpoint.states ?? 0)} estados`
      checkpointSelect.append(option)
      if (checkpoint.active) checkpointSelect.value = option.value
    }
    if (!checkpoints.length) {
      checkpointSelect.append(new Option('Se creará al terminar el primer episodio', ''))
    }
    checkpointSelect.disabled = !checkpoints.length
    activateCheckpointButton.disabled = !checkpoints.length
    snapshotCheckpointButton.disabled = !checkpoints.length
    const selected = checkpoints.find((item) => item.active) ?? checkpoints[0]
    checkpointMeta.textContent = selected
      ? `${Number(selected.episodes ?? 0)} episodios · ${Number(selected.updates ?? 0)} actualizaciones · ε ${Number(selected.epsilon ?? 0).toFixed(3)} · ${String(selected.updated_at ?? 'fecha desconocida')}`
      : 'Todavía no existe memoria persistida. El primer episodio terminado creará el checkpoint.'
    if (isRecord(payload) && payload.persistent === false) {
      checkpointMeta.textContent += ' · servidor sin disco persistente configurado'
    }
  } catch (error) {
    checkpointMeta.textContent = errorMessage(error)
  } finally {
    refreshCheckpointsButton.disabled = !apiOnline
  }
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
  if (!manualSessionId) manualButton.textContent = `Entrenar Q-learning en L${level}`
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
  displayedEpisode = episode
  currentMap = map
  updateReferenceMap(map)
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
    refreshCheckpointsButton.disabled = false
    if (normalizedCatalog.runs.length) hud.hideNotice()
    else hud.showNotice('Catálogo vacío', 'La API responde, pero /catalog todavía no contiene modelos recuperables.')
    void refreshMonitor()
    void refreshEpisodes()
    void api.qlearningStatus().then(showQTraining).catch(() => undefined)
    void refreshCheckpoints()
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
    const [payload, metricsPayload, qTrainingPayload] = await Promise.all([
      api.episodes(100),
      api.episodeMetrics().catch(() => undefined),
      api.qlearningStatus().catch(() => undefined),
    ])
    if (qTrainingPayload) showQTraining(qTrainingPayload)
    const episodes = isRecord(payload) && Array.isArray(payload.episodes) ? payload.episodes.filter(isRecord) : []
    const storage = isRecord(payload) && isRecord(payload.storage) ? payload.storage : undefined
    required<HTMLElement>('#experimentStorageMeta').textContent = storage
      ? `${Number(storage.retained_experiments ?? 0)}/${Number(storage.max_experiments ?? 10)} replays seleccionados · ${Number(storage.protected_experiments ?? 0)} protegidos · ${Number(storage.summarized_episodes ?? 0)} episodios resumidos${Number(storage.historical_untouched ?? 0) ? ` · ${Number(storage.historical_untouched)} históricos intactos` : ''}`
      : 'Se conservan hasta 10 replays completos; las métricas resumidas continúan para todo el entrenamiento.'
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
      duration_s: Number(item.duration_s ?? 0),
      end: String(item.end ?? 'en progreso'),
    })).filter((item) => Number.isInteger(item.episode_number) && item.episode_number > 0)
    const finished = summaries.filter((episode) => episode.success !== null)
    trainingDashboard.setEpisodes(finished)
    const metrics = isRecord(metricsPayload) ? metricsPayload : undefined
    const totalEpisodes = Number(metrics?.episodes ?? summaries.length)
    const successRate = metrics ? Number(metrics.success_rate ?? 0) * 100 : (finished.length
      ? (finished.filter((episode) => episode.success).length / finished.length) * 100
      : 0)
    learningMeta.textContent = totalEpisodes
      ? `${totalEpisodes} episodios entrenados · ${successRate.toFixed(1)}% de éxito · recompensa media ${Number(metrics?.reward_mean ?? 0).toFixed(1)} · ${Number(metrics?.samples_collected ?? 0)} muestras acumuladas`
      : 'Sin episodios registrados.'
    episodeList.replaceChildren()
    if (!episodes.length) {
      const empty = document.createElement('p')
      empty.className = 'section-copy'
      empty.textContent = 'Aún no hay episodios planetarios guardados. Inicia Q-learning para crear el primero.'
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
      const model = document.createElement('small')
      model.className = 'episode-model-line'
      const modelName = String(item.policy_version ?? item.model_name ?? 'planetary-qlearning')
      const modelStates = Number(latestQTraining?.states ?? 0)
      const modelConnections = modelStates * 7
      model.textContent = `Modelo ${modelName} · 7 acciones · memoria asociada: ${modelStates.toLocaleString('es-CO')} estados / ${modelConnections.toLocaleString('es-CO')} conexiones · ${Number(latestQTraining?.updates ?? 0).toLocaleString('es-CO')} actualizaciones`
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
      button.append(numberLabel, description, status, protect, model)
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

const scheduleAutomaticStep = (delay = 150) => {
  window.clearTimeout(automaticTimer)
  if (automaticRunning && manualSessionId) {
    automaticTimer = window.setTimeout(() => { void sendAutomaticAction() }, delay)
  }
}

const startManualMode = async () => {
  if (!api) return
  window.clearTimeout(automaticTimer)
  automaticRunning = false
  manualButton.disabled = true
  manualButton.textContent = 'Preparando agente…'
  try {
    const environment = manualEnvironment.value === 'planetary' ? 'planetary' : 'beacon'
    const difficulty = Number(difficultySelect.value)
    const seed = environment === 'planetary' ? randomScenarioSeed() : 0
    const policy = environment === 'planetary' ? 'planetary-qlearning' : 'baseline-rule'
    const response = await api.reset(levelSelect.value, environment, difficulty, seed, policy)
    const frame = frameFromResponse(response)
    if (!response.session_id) throw new Error('/sim/reset no devolvió session_id.')
    if (!frame) throw new Error('/sim/reset no devolvió un cuadro compatible.')
    manualSessionId = response.session_id
    const map = response.map ? normalizeMap(response.map) : currentMap
    missionSeedValue.textContent = environment === 'planetary'
      ? `semilla ${String(response.seed ?? seed)} · ${map.scenario_id ?? 'escenario procedural'}`
      : 'modo de balizas'
    const initial = normalizeEpisode(
      { ...response, frames: [frame], episode: `auto-${automaticEpisodeCount + 1}`, end: 'running' },
      {
        mode: 'entrenamiento-replay',
        runId: 'planetary-qlearning',
        modelName: 'Q-learning tabular persistente',
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
    automaticEpisodeCount += 1
    loadEpisode(manualEpisode, map, 'api', 'API · Q-LEARNING')
    const conditions = map.environment
    required<HTMLElement>('#planetReadoutTitle').textContent = `${String(conditions?.planet?.planet_id ?? 'MARTE').toUpperCase()} · ${map.difficulty ?? `L${difficulty}`}`
    required<HTMLElement>('#readoutGravity').textContent = `${Number(conditions?.planet?.gravity_m_s2 ?? 3.71).toFixed(2)} m/s²`
    required<HTMLElement>('#readoutWind').textContent = `${Number(conditions?.wind_speed_m_s ?? 0).toFixed(1)} m/s`
    required<HTMLElement>('#readoutTemperature').textContent = `${Number(conditions?.temperature_c ?? -55).toFixed(0)} °C`
    required<HTMLElement>('#readoutVisibility').textContent = `${Number(conditions?.visibility_m ?? 0).toFixed(1)} m`
    required<HTMLElement>('#readoutStorm').textContent = difficulty >= 3 ? 'ciclos dinámicos' : 'sin tormenta'
    required<HTMLElement>('#readoutTraction').textContent = Number(conditions?.traction ?? 0).toFixed(2)
    required<HTMLElement>('#readoutSamples').textContent = `${map.mission?.requested_samples ?? 0} × ${map.mission?.target_type ?? '—'}`
    setProvenance('planetary-qlearning', environment === 'planetary' ? `memoria persistente · escenario procedural · L${difficulty}` : 'baseline autónomo')
    automaticNetworkRetries = 0
    automaticRunning = true
    setManualControls(false)
    manualButton.textContent = 'Pausar entrenamiento'
    hud.hideNotice()
    scheduleAutomaticStep(400)
  } catch (error) {
    manualSessionId = undefined
    manualEpisode = undefined
    setManualControls(false)
    hud.showNotice('No se pudo iniciar Q-learning', errorMessage(error))
  } finally {
    manualButton.disabled = false
    if (!manualSessionId) manualButton.textContent = `Entrenar Q-learning en L${difficultySelect.value}`
  }
}

// Cada petición cruza la red (~0,45 s con Render); pedir un lote de pasos y
// animarlo mientras llega el siguiente evita que el rover avance a saltos lentos.
const AUTOMATIC_BATCH = 16

async function sendAutomaticAction() {
  if (!api || !manualSessionId || !manualEpisode || manualBusy) return
  manualBusy = true
  let retryDelay: number | undefined
  let nextDelay = 0
  setManualControls(false)
  try {
    const shownPosition = scene.position
    const response = await api.step(manualSessionId, undefined, AUTOMATIC_BATCH)
    automaticNetworkRetries = 0
    const frame = frameFromResponse(response)
    if (!frame) throw new Error('/sim/step no devolvió un cuadro compatible.')
    const batch = Array.isArray(response.frames)
      ? response.frames.map((item) => normalizeFrame(item)).filter((item): item is ReplayFrame => Boolean(item))
      : []
    const frames = [...manualEpisode.frames, ...(batch.length ? batch : [frame])]
    if (typeof response.storm_intensity === 'number') {
      required<HTMLElement>('#readoutStorm').textContent = response.storm_intensity > 0.05
        ? `${Math.round(response.storm_intensity * 100)}% · ACTIVA`
        : 'en calma'
    }
    showQTraining(response.training)
    const finished = Boolean(response.terminated || response.truncated)
    const map = response.map ? normalizeMap(response.map) : currentMap
    // La etapa 3 es siempre "volver a la base": en N1 se pasa de 1 a 3, con una sola baliza.
    const visitedNow = Math.min(frame[FRAME_INDEX.stage] - 1, map.beacons.length)
    manualEpisode = {
      ...manualEpisode,
      frames,
      steps: frames.length - 1,
      maxSteps: typeof response.max_steps === 'number' ? response.max_steps : manualEpisode.maxSteps,
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
    loadEpisode(manualEpisode, map, 'api', 'API · Q-LEARNING')
    // Continúa la animación desde donde iba (loadEpisode la reinicia) hasta el último cuadro.
    scene.seek(Math.min(shownPosition, frames.length - 1))
    scene.setPlaying(true)
    const backlogSeconds = (frames.length - 1 - scene.position) / scene.framesPerSecond
    // Pide el siguiente lote cuando quede medio lote por animar: llega justo a tiempo.
    nextDelay = Math.max(0, (backlogSeconds - AUTOMATIC_BATCH / 2 / scene.framesPerSecond) * 1000)
    if (finished) {
      manualSessionId = undefined
      const reachedRewardFloor = response.end === 'limite_recompensa'
      hud.showNotice(
        reachedRewardFloor
            ? 'Límite de −20 alcanzado'
            : 'Episodio automático finalizado',
        reachedRewardFloor
            ? 'La transición negativa se guardó en la tabla Q. El agente iniciará otro escenario con esa memoria actualizada.'
            : `Causa: ${manualEpisode.end}. El agente generará un escenario nuevo.`,
      )
      void refreshEpisodes()
      if (automaticRunning) automaticTimer = window.setTimeout(() => { void startManualMode() }, backlogSeconds * 1000 + 1200)
    }
  } catch (error) {
    const transientNetworkFailure = error instanceof ApiError && error.status === undefined
    if (transientNetworkFailure && automaticRunning && automaticNetworkRetries < 6) {
      automaticNetworkRetries += 1
      retryDelay = Math.min(5000, 500 * 2 ** (automaticNetworkRetries - 1))
      hud.showNotice(
        'Reconectando el entrenamiento',
        `La API no respondió. Reintento ${automaticNetworkRetries}/6 en ${(retryDelay / 1000).toFixed(1)} s; el checkpoint ya guardado se conserva.`,
      )
    } else {
      automaticRunning = false
      automaticNetworkRetries = 0
      hud.showNotice('El entrenamiento Q-learning se detuvo', errorMessage(error))
    }
  } finally {
    manualBusy = false
    setManualControls(false)
    if (manualSessionId && automaticRunning) scheduleAutomaticStep(retryDelay ?? nextDelay)
  }
}

function setManualControls(enabled: boolean) {
  keypadButtons.forEach((button) => {
    button.disabled = !enabled || (Number(button.dataset.action) === 5 && manualEnvironment.value !== 'planetary')
  })
}

function metricNumber(record: UnknownRecord | undefined, key: string): number {
  return record && typeof record[key] === 'number' ? Number(record[key]) : 0
}

const ppoText = (selector: string, value: string) => { required<HTMLElement>(selector).textContent = value }

function selectPpoLevel(item: UnknownRecord) {
  ppoRunning = false
  window.clearTimeout(ppoTimer)
  ppoPause.disabled = true
  selectedPpoLevel = Number(item.level ?? 1)
  const model = isRecord(item.model) ? item.model : {}
  selectedPpoRun = String(model.run_id ?? '')
  ppoTriedRuns = new Set(selectedPpoRun ? [selectedPpoRun] : [])
  ppoText('#ppoMode', 'CHECKPOINT SELECCIONADO · SIN EJECUTAR')
  ppoText('#ppoAction', '—')
  ppoText('#ppoValue', '—')
  ppoText('#ppoSceneLevel', `L${selectedPpoLevel}`)
  ppoText('#ppoCheckpoint', selectedPpoRun.slice(0, 12) || '—')
  ppoModelSwitch.textContent = `L${selectedPpoLevel} usa el checkpoint ${selectedPpoRun.slice(0, 12) || 'no disponible'}. Si falla y hay alternativa, el visor lo indicará antes de reintentar.`
  ppoStart.disabled = !apiOnline || !selectedPpoRun
  ppoRestart.disabled = !selectedPpoRun
  showPpoEvidence(item)
}

function renderPpoProbabilities(decision: UnknownRecord | undefined, actions: string[]) {
  const probabilities = Array.isArray(decision?.probabilities) ? decision.probabilities : []
  const selected = Number(decision?.action ?? -1)
  ppoActionProbabilities.replaceChildren()
  probabilities.forEach((raw, index) => {
    const probability = Number(raw)
    const row = document.createElement('div')
    row.className = `ppo-probability${index === selected ? ' selected' : ''}`
    row.innerHTML = `<i style="width:${Math.max(0, Math.min(100, probability * 100))}%"></i><span><b>${actions[index] ?? `acción ${index}`}</b><em>${(probability * 100).toFixed(1)}%</em></span>`
    ppoActionProbabilities.append(row)
  })
}

function renderPpoDecisionHistory() {
  ppoDecisionHistory.replaceChildren()
  for (const item of ppoDecisionRows.slice(-100).reverse()) {
    const row = document.createElement('tr')
    const values = [item.step, item.action_name, `${(Number(item.probability ?? 0) * 100).toFixed(1)}%`,
      Number(item.value ?? 0).toFixed(2), Number(item.reward ?? 0).toFixed(2), item.event ?? '—']
    for (const value of values) {
      const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell)
    }
    ppoDecisionHistory.append(row)
  }
}

function updatePpoLive(response: SimulationResponse, frame: ReplayFrame, map: ExpeditionMap) {
  const decision = isRecord(response.decision) ? response.decision : undefined
  const actions = map.actions
  const action = Number(response.action ?? frame[FRAME_INDEX.action])
  const probability = Array.isArray(decision?.probabilities) ? Number(decision.probabilities[action] ?? 0) : 0
  const rays = frame[FRAME_INDEX.rays].filter((distance) => distance > 0)
  ppoText('#ppoStep', String(response.steps ?? Math.max(0, (ppoEpisode?.frames.length ?? 1) - 1)))
  ppoText('#ppoReward', Number(response.return ?? 0).toFixed(2))
  ppoText('#ppoBattery', `${Number(response.battery ?? frame[FRAME_INDEX.battery]).toFixed(1)}%`)
  ppoText('#ppoEnd', String(response.end ?? 'en curso'))
  ppoText('#ppoAction', actions[action] ?? `acción ${action}`)
  ppoText('#ppoValue', decision ? Number(decision.value ?? 0).toFixed(3) : 'no disponible')
  ppoText('#ppoSamples', `${Number(response.samples_collected ?? 0)} / ${Number(response.requested_samples ?? map.mission?.requested_samples ?? 0)}`)
  ppoText('#ppoCollisions', String(response.collisions ?? 0))
  ppoText('#ppoNearestObstacle', rays.length ? `${Math.min(...rays).toFixed(2)} m` : 'fuera de alcance')
  ppoText('#ppoObjective', Number(response.samples_collected ?? 0) >= Number(response.requested_samples ?? map.mission?.requested_samples ?? 0)
    ? 'Regresar a la base' : `Recolectar ${String(response.target_type ?? map.mission?.target_type ?? 'muestras')}`)
  renderPpoProbabilities(decision, actions)
  if (decision) {
    ppoDecisionRows.push({ step: response.steps, action_name: actions[action] ?? action, probability,
      value: decision.value, reward: response.reward, event: response.event })
    renderPpoDecisionHistory()
  }
}

const schedulePpoStep = () => {
  window.clearTimeout(ppoTimer)
  if (ppoRunning && ppoSessionId) {
    const delay = Number(required<HTMLSelectElement>('#ppoSpeed').value)
    ppoTimer = window.setTimeout(() => { void stepPpoEpisode() }, delay)
  }
}

async function startPpoEpisode() {
  if (!api || !selectedPpoRun) return
  window.clearTimeout(ppoTimer)
  ppoRunning = false
  ppoStart.disabled = true
  ppoStart.textContent = 'Cargando checkpoint…'
  try {
    const seed = 910000 + selectedPpoLevel
    const response = await api.reset('n2', 'planetary', selectedPpoLevel, seed, `trained:${selectedPpoRun}`, ppoSessionId)
    const frame = frameFromResponse(response)
    if (!response.session_id || !frame || !response.map) throw new Error('La API PPO no devolvió sesión, mapa y frame inicial.')
    ppoSessionId = response.session_id
    ppoMap = normalizeMap(response.map)
    ppoEpisode = normalizeEpisode({ ...response, frames: [frame], episode: `PPO-L${selectedPpoLevel}` }, {
      mode: 'inferencia PPO', runId: selectedPpoRun, modelName: 'PPO checkpoint aprobado',
      environmentVersion: environmentVersionFrom(response) ?? apiEnvironmentVersion, levelId: `L${selectedPpoLevel}`,
    })
    ppoEpisode = { ...ppoEpisode, scenarioId: ppoMap.scenario_id, planet: ppoMap.environment?.planet?.planet_id,
      objective: `Recolectar ${ppoMap.mission?.requested_samples ?? 0} × ${ppoMap.mission?.target_type ?? 'muestra'}`,
      requestedSamples: ppoMap.mission?.requested_samples, samplesCollected: 0 }
    ppoDecisionRows = []
    const policyModel = isRecord(response.policy_model) ? response.policy_model : {}
    const architecture = isRecord(policyModel.architecture) ? policyModel.architecture : {}
    ppoText('#ppoArchitectureNote', `Arquitectura real: ${Array.isArray(architecture.policy) ? architecture.policy.join(' → ') : '132 → 64 → 64 → 7'} · ${String(architecture.activation ?? 'Tanh')} · ${Number(architecture.actors ?? 1)} actor(es) especializados. Rama de valor ${Array.isArray(architecture.value) ? architecture.value.join(' → ') : '132 → 64 → 64 → 1'}. Hash ${String(policyModel.model_sha256 ?? '').slice(0, 12)}…. Las activaciones se resumen sin atribuirles una explicación causal.`)
    ppoScene.setMap(ppoMap); ppoScene.setEpisode(ppoEpisode); ppoScene.seek(0)
    ppoText('#ppoMode', 'INFERENCIA PPO REAL')
    ppoText('#ppoObjective', ppoEpisode.objective ?? 'Misión planetaria')
    ppoText('#ppoStep', '0'); ppoText('#ppoReward', '0,00'); ppoText('#ppoBattery', '100%'); ppoText('#ppoEnd', 'En curso')
    ppoRunning = true
    ppoPause.disabled = false; ppoPause.textContent = 'Pausar'; ppoRestart.disabled = false
    schedulePpoStep()
  } catch (error) {
    ppoText('#ppoEnd', 'Error')
    ppoText('#ppoObjective', errorMessage(error))
  } finally {
    ppoStart.textContent = 'Iniciar inferencia'
    ppoStart.disabled = !apiOnline || !selectedPpoRun
  }
}

async function stepPpoEpisode() {
  if (!api || !ppoSessionId || !ppoEpisode || !ppoMap || !ppoRunning) return
  try {
    const response = await api.step(ppoSessionId)
    const frame = frameFromResponse(response)
    if (!frame) throw new Error('La inferencia PPO no devolvió un frame válido.')
    const frames = [...ppoEpisode.frames, frame]
    ppoEpisode = { ...ppoEpisode, frames, steps: frames.length - 1, reward: Number(response.return ?? ppoEpisode.reward),
      battery: frame[FRAME_INDEX.battery], success: Boolean(response.success), end: String(response.end ?? 'running'),
      samplesCollected: Number(response.samples_collected ?? ppoEpisode.samplesCollected ?? 0),
      missionTimeline: [...(ppoEpisode.missionTimeline ?? []), { target_type: String(response.target_type ?? ppoMap.mission?.target_type ?? ''),
        requested: Number(response.requested_samples ?? ppoMap.mission?.requested_samples ?? 0),
        collected: Number(response.samples_collected ?? 0), inventory: Array.isArray(response.inventory) ? response.inventory.filter((x): x is string => typeof x === 'string') : [],
        samples: ppoMap.samples, event: typeof response.event === 'string' ? response.event : null, status: String(response.end ?? 'running') }] }
    ppoScene.setEpisode(ppoEpisode); ppoScene.seek(frames.length - 1)
    updatePpoLive(response, frame, ppoMap)
    if (response.terminated || response.truncated) {
      ppoRunning = false; ppoPause.disabled = true; ppoText('#ppoEnd', String(response.end ?? 'finalizado'))
      const alternatives = ppoCampaignModels
        .filter((candidate) => candidate.level >= 6 && candidate.runId !== selectedPpoRun && !ppoTriedRuns.has(candidate.runId))
        .sort((a, b) => b.level - a.level)
      const next = !response.success && selectedPpoLevel >= 6 ? alternatives[0] : undefined
      if (next) {
        const previousRun = selectedPpoRun
        selectedPpoRun = next.runId
        ppoTriedRuns.add(next.runId)
        ppoText('#ppoCheckpoint', selectedPpoRun.slice(0, 12))
        ppoText('#ppoMode', `CAMBIO AUTOMÁTICO · PPO L${selectedPpoLevel}`)
        ppoText('#ppoObjective', `El checkpoint anterior terminó sin éxito; se reintenta L${selectedPpoLevel} con otro modelo.`)
        ppoModelSwitch.textContent = `Cambio automático en L${selectedPpoLevel}: ${previousRun.slice(0, 12)} no llegó a la meta (${String(response.end ?? 'sin éxito')}). Se prueba ${selectedPpoRun.slice(0, 12)}. Es un reintento alternativo, no una validación.`
        window.setTimeout(() => { void startPpoEpisode() }, 700)
      } else if (!response.success && selectedPpoLevel >= 6) {
        ppoModelSwitch.textContent = `L${selectedPpoLevel}: se agotaron los checkpoints alternativos disponibles sin éxito. El resultado se mantiene como fallo; no se presenta como misión superada.`
      }
    } else schedulePpoStep()
  } catch (error) {
    ppoRunning = false; ppoText('#ppoEnd', 'Detenido'); ppoText('#ppoObjective', errorMessage(error))
  }
}

function showPpoEvidence(item: UnknownRecord) {
  const scenario = isRecord(item.scenario) ? item.scenario : {}
  const model = isRecord(item.model) ? item.model : {}
  const evidence = isRecord(item.evidence) ? item.evidence : {}
  const metrics = isRecord(evidence.metrics) ? evidence.metrics : {}
  const samples = Array.isArray(scenario.samples) ? scenario.samples : []
  const targets = samples.filter((sample) => isRecord(sample) && sample.target === true).length
  const distractors = Math.max(0, samples.length - targets)
  const success = metricNumber(metrics, 'success_rate') * 100
  const collisions = metricNumber(metrics, 'collisions_per_episode')
  const reward = metricNumber(metrics, 'return_mean')
  ppoEvidenceDetail.innerHTML = `
    <div><span class="eyebrow">Escenario fijo</span><strong>L${Number(item.level)} · ${String(scenario.name ?? 'Sin nombre')}</strong></div>
    <dl>
      <div><dt>Checkpoint PPO</dt><dd>${String(model.run_id ?? '').slice(0, 10)}</dd></div>
      <div><dt>Entrenamiento</dt><dd>${Number(model.total_timesteps ?? 0).toLocaleString('es-CO')} pasos</dd></div>
      <div><dt>Evaluación</dt><dd>${Number(evidence.cases ?? 0)} episodios · ${success.toFixed(0)}% éxito</dd></div>
      <div><dt>Resultado medio</dt><dd>${reward.toFixed(1)} recompensa · ${collisions.toFixed(2)} choques</dd></div>
      <div><dt>Misión</dt><dd>${targets} objetivos · ${distractors} distractores · ${String(scenario.target_type ?? 'muestra')}</dd></div>
      <div><dt>Complejidad</dt><dd>${Array.isArray(scenario.rocks) ? scenario.rocks.length : 0} rocas · ${Array.isArray(scenario.craters) ? scenario.craters.length : 0} cráteres · viento ${Number(scenario.wind_speed_m_s ?? 0)} m/s</dd></div>
    </dl>`
}

async function loadPpoCampaign() {
  if (!api) return
  try {
    const payload = await api.ppoCampaign()
    if (!isRecord(payload) || !Array.isArray(payload.levels)) throw new Error('La API no devolvió la campaña PPO.')
    const levels = payload.levels.filter(isRecord).sort((a, b) => {
      const levelDifference = Number(a.level ?? 0) - Number(b.level ?? 0)
      if (levelDifference !== 0) return levelDifference
      const aModel = isRecord(a.model) ? a.model : {}
      const bModel = isRecord(b.model) ? b.model : {}
      return Number(aModel.total_timesteps ?? 0) - Number(bModel.total_timesteps ?? 0)
    })
    ppoCampaignModels = levels.map((item) => {
      const model = isRecord(item.model) ? item.model : {}
      return { level: Number(item.level ?? 0), runId: String(model.run_id ?? '') }
    }).filter((candidate) => candidate.level >= 1 && Boolean(candidate.runId))
    ppoCampaignMeta.textContent = `${levels.length} checkpoints registrados · ${Number(payload.latest_total_timesteps ?? 0).toLocaleString('es-CO')} pasos acumulados en el hito final · la evidencia histórica se distingue de la inferencia en vivo`
    ppoCampaignLevels.replaceChildren()
    for (const item of levels) {
      const model = isRecord(item.model) ? item.model : {}
      const scenario = isRecord(item.scenario) ? item.scenario : {}
      const evidence = isRecord(item.evidence) ? item.evidence : {}
      const metrics = isRecord(evidence.metrics) ? evidence.metrics : {}
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'ppo-level-card'
      button.innerHTML = `<span>L${Number(item.level)}</span><strong>${String(scenario.name ?? 'Escenario')}</strong><small>${Number(model.total_timesteps ?? 0).toLocaleString('es-CO')} pasos</small><i>${(metricNumber(metrics, 'success_rate') * 100).toFixed(0)}% histórico</i>`
      button.addEventListener('click', () => {
        ppoCampaignLevels.querySelectorAll('button').forEach((node) => node.classList.remove('selected'))
        button.classList.add('selected')
        selectPpoLevel(item)
        void startPpoEpisode()
      })
      ppoCampaignLevels.append(button)
    }
    const best = levels.at(-1)
    if (best) {
      ppoCampaignLevels.querySelector('button:last-child')?.classList.add('selected')
      selectPpoLevel(best)
      await startPpoEpisode()
    }
  } catch (error) {
    ppoCampaignMeta.textContent = `No se pudo cargar el modelado PPO: ${errorMessage(error)}`
  }
}

ppoStart.addEventListener('click', () => { void startPpoEpisode() })
ppoRestart.addEventListener('click', () => { void startPpoEpisode() })
ppoPause.addEventListener('click', () => {
  ppoRunning = !ppoRunning
  ppoPause.textContent = ppoRunning ? 'Pausar' : 'Reanudar'
  if (ppoRunning) schedulePpoStep()
  else window.clearTimeout(ppoTimer)
})
const ppoViewport = required<HTMLElement>('.ppo-viewport-wrap')
ppoFullscreen.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement === ppoViewport) await document.exitFullscreen()
    else await ppoViewport.requestFullscreen()
  } catch {
    ppoViewport.classList.toggle('ppo-expanded')
    const active = ppoViewport.classList.contains('ppo-expanded')
    ppoFullscreen.setAttribute('aria-pressed', String(active))
    ppoFullscreen.textContent = active ? 'Salir de pantalla completa' : 'Pantalla completa'
  }
})
document.addEventListener('fullscreenchange', () => {
  const active = document.fullscreenElement === ppoViewport
  ppoFullscreen.setAttribute('aria-pressed', String(active))
  ppoFullscreen.textContent = active ? 'Salir de pantalla completa' : 'Pantalla completa'
})
required<HTMLButtonElement>('#ppoFollow').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement
  const active = button.getAttribute('aria-pressed') !== 'true'
  button.setAttribute('aria-pressed', String(active)); ppoScene.setFollow(active)
})
required<HTMLButtonElement>('#ppoTrail').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement
  const active = button.getAttribute('aria-pressed') !== 'true'
  button.setAttribute('aria-pressed', String(active)); ppoScene.setTrailVisible(active)
})
required<HTMLButtonElement>('#ppoSensors').addEventListener('click', (event) => {
  const button = event.currentTarget as HTMLButtonElement
  const active = button.getAttribute('aria-pressed') !== 'true'
  button.setAttribute('aria-pressed', String(active)); ppoScene.setSensorsVisible(active)
})
document.querySelectorAll<HTMLButtonElement>('[data-ppo-tab]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll<HTMLButtonElement>('[data-ppo-tab]').forEach((item) => item.setAttribute('aria-selected', String(item === button)))
    const selected = button.dataset.ppoTab
    for (const [name, selector] of Object.entries({ simulator: '#ppoTabSimulator', model: '#ppoTabModel', decisions: '#ppoTabDecisions', experiments: '#ppoTabExperiments', metrics: '#ppoTabMetrics' })) {
      required<HTMLElement>(selector).hidden = name !== selected
    }
    if (selected === 'metrics') trainingDashboard.resize()
  })
})
required<HTMLButtonElement>('#ppoGoExperiments').addEventListener('click', () => {
  document.querySelector('.episode-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
})

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

const fullscreenButton = required<HTMLButtonElement>('#fullscreenButton')
const viewportPanel = required<HTMLElement>('.viewport-panel')
fullscreenButton.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen()
    else await viewportPanel.requestFullscreen()
  } catch (error) {
    hud.showNotice('No fue posible cambiar la pantalla', errorMessage(error))
  }
})
document.addEventListener('fullscreenchange', () => {
  const active = document.fullscreenElement === viewportPanel
  fullscreenButton.setAttribute('aria-pressed', String(active))
  fullscreenButton.textContent = active ? 'Salir de pantalla completa' : 'Pantalla completa'
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
manualButton.addEventListener('click', () => {
  if (!manualSessionId) {
    void startManualMode()
    return
  }
  automaticRunning = !automaticRunning
  manualButton.textContent = automaticRunning ? 'Pausar entrenamiento' : 'Reanudar entrenamiento'
  if (automaticRunning) scheduleAutomaticStep(0)
  else window.clearTimeout(automaticTimer)
})
for (const button of planetLevelButtons) {
  button.addEventListener('click', () => {
    difficultySelect.value = button.dataset.planetLevel ?? '1'
    manualEnvironment.value = 'planetary'
    updatePlanetaryLevel()
    document.querySelector('#planetaryMission')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    if (apiOnline) void startManualMode()
  })
}
difficultySelect.addEventListener('change', () => {
  updatePlanetaryLevel()
  if (apiOnline) void startManualMode()
})
manualEnvironment.addEventListener('change', () => {
  difficultySelect.disabled = manualEnvironment.value !== 'planetary'
  setManualControls(false)
  manualButton.textContent = `Entrenar Q-learning en L${difficultySelect.value}`
})
refreshEpisodesButton.addEventListener('click', () => { void refreshEpisodes() })
refreshCheckpointsButton.addEventListener('click', () => { void refreshCheckpoints() })
checkpointSelect.addEventListener('change', async () => {
  if (!api || !checkpointSelect.value) return
  const payload = await api.qlearningCheckpoints().catch(() => undefined)
  const checkpoints = isRecord(payload) && Array.isArray(payload.checkpoints) ? payload.checkpoints.filter(isRecord) : []
  const selected = checkpoints.find((item) => item.name === checkpointSelect.value)
  if (selected) checkpointMeta.textContent = `${Number(selected.episodes ?? 0)} episodios · ${Number(selected.updates ?? 0)} actualizaciones · ε ${Number(selected.epsilon ?? 0).toFixed(3)} · ${String(selected.updated_at ?? 'fecha desconocida')}`
})
activateCheckpointButton.addEventListener('click', async () => {
  if (!api || !checkpointSelect.value) return
  activateCheckpointButton.disabled = true
  try {
    const result = await api.activateQlearningCheckpoint(checkpointSelect.value)
    showQTraining(result)
    checkpointMeta.textContent = 'Memoria activada. El siguiente episodio continuará desde ese checkpoint.'
    manualSessionId = undefined
    automaticRunning = false
    manualButton.textContent = `Entrenar Q-learning en L${difficultySelect.value}`
    await refreshCheckpoints()
  } catch (error) {
    hud.showNotice('No se pudo activar el checkpoint', errorMessage(error))
  } finally {
    activateCheckpointButton.disabled = !apiOnline || !checkpointSelect.value
  }
})
snapshotCheckpointButton.addEventListener('click', async () => {
  if (!api || !checkpointSelect.value) return
  snapshotCheckpointButton.disabled = true
  try {
    const label = `memoria-ep-${Date.now()}`
    await api.snapshotQlearningCheckpoint(label)
    await refreshCheckpoints()
    checkpointMeta.textContent = 'Copia guardada. Puedes volver a activarla cuando quieras.'
  } catch (error) {
    hud.showNotice('No se pudo guardar la copia', errorMessage(error))
  } finally {
    snapshotCheckpointButton.disabled = !apiOnline || !checkpointSelect.value
  }
})
loadEpisodeButton.addEventListener('click', () => { void loadStoredEpisode(Number(episodeNumberInput.value)) })
episodeNumberInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') void loadStoredEpisode(Number(episodeNumberInput.value))
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

difficultySelect.value = '1'
updatePlanetaryLevel()
mountAgentTraining({
  baseUrl: () => api?.baseUrl,
  level: () => Number(difficultySelect.value),
  replay: (payload, label) => loadReplay(payload, label),
  frame: (payload, label) => {
    if (!isRecord(payload)) throw new Error('La API no devolvió una evaluación PPO compatible.')
    const map = normalizeMap(payload.map)
    const episode = normalizeEpisode(payload, {
      mode: 'evaluación PPO',
      runId: String(payload.run_id ?? 'sin-run'),
      modelName: 'PPO',
      levelId: `l${difficultySelect.value}`,
    })
    loadEpisode(episode, map, 'api', label)
    setProvenance('PPO', 'evaluación de versión guardada')
  },
})
void connectApi().then(() => {
  if (apiOnline) {
    void loadPpoCampaign()
    if (!manualSessionId) void startManualMode()
  }
})
