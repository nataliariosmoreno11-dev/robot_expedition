import type {
  CatalogData,
  EpisodeData,
  EpisodeRock,
  ExpeditionMap,
  HistoryPoint,
  ReplayBundle,
  ReplayFrame,
  RunOption,
  SampleSpec,
  MissionState,
} from './types'

type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const asRecord = (value: unknown): UnknownRecord => (isRecord(value) ? value : {})

const asNumber = (value: unknown, fallback = 0): number => {
  const parsed = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim() ? value : fallback

const asBoolean = (value: unknown, fallback = false): boolean =>
  typeof value === 'boolean' ? value : fallback

const numberArray = (value: unknown): number[] =>
  Array.isArray(value) ? value.map((item) => asNumber(item)) : []

const tuple = <T extends number[]>(value: unknown, fallback: T): T => {
  if (!Array.isArray(value) || value.length < fallback.length) return fallback
  return value.slice(0, fallback.length).map((item) => asNumber(item)) as T
}

const normalizeSamples = (value: unknown): SampleSpec[] | undefined => {
  if (!Array.isArray(value)) return undefined
  return value.filter(isRecord).map((item, index) => ({
    sample_id: asString(item.sample_id, `S${index + 1}`),
    sample_type: asString(item.sample_type, 'desconocida'),
    x: asNumber(item.x),
    y: asNumber(item.y),
    target: asBoolean(item.target),
    collected: item.collected === undefined ? undefined : asBoolean(item.collected),
  }))
}

const normalizeMissionTimeline = (value: unknown): MissionState[] | undefined => {
  if (!Array.isArray(value)) return undefined
  return value.map((entry) => {
    const item = asRecord(entry)
    return {
      target_type: asString(item.target_type) || undefined,
      requested: item.requested === undefined ? undefined : asNumber(item.requested),
      collected: item.collected === undefined ? undefined : asNumber(item.collected),
      inventory: Array.isArray(item.inventory) ? item.inventory.map((id) => asString(id)).filter(Boolean) : undefined,
      samples: normalizeSamples(item.samples),
      event: typeof item.event === 'string' ? item.event : null,
      status: asString(item.status) || undefined,
    }
  })
}

export const isReplayFrame = (value: unknown): value is ReplayFrame =>
  Array.isArray(value) &&
  value.length >= 13 &&
  value.slice(0, 12).every((item) => Number.isFinite(Number(item))) &&
  Array.isArray(value[12])

export const normalizeFrame = (value: unknown): ReplayFrame | undefined => {
  if (!isReplayFrame(value)) return undefined
  return [
    asNumber(value[0]),
    asNumber(value[1]),
    asNumber(value[2]),
    asNumber(value[3], 1),
    asNumber(value[4]),
    asNumber(value[5]),
    asNumber(value[6]),
    Math.max(1, asNumber(value[7], 1)),
    asNumber(value[8], 100),
    asNumber(value[9], -1),
    asNumber(value[10]),
    asNumber(value[11]),
    numberArray(value[12]),
  ]
}

const buildTerrain = (n = 61): number[] => {
  const values: number[] = []
  for (let row = 0; row < n; row += 1) {
    const y = 6 - (row / (n - 1)) * 12
    for (let column = 0; column < n; column += 1) {
      const x = (column / (n - 1)) * 12
      const dune = x > 5.5 && x < 8.8 ? 0.16 * Math.sin((x - 5.5) * 2.2) * Math.cos(y * 1.15) : 0
      const crack = Math.abs(x - 3.5) < 0.5 && Math.abs(y - 3.5) < 0.22 ? -0.5 : 0
      values.push(Number((dune + crack).toFixed(3)))
    }
  }
  return values
}

export const createContractMap = (): ExpeditionMap => ({
  size: [12, 12],
  base: [1.5, 0, 1, 1],
  beacons: [[9.5, 3.5], [9.5, -3.5]],
  visit_radius: 0.6,
  crack: [3.5, 3.5, 0.5, 0.2],
  crack_depth: 0.5,
  sand: [7, -4.5, 1, 0.7],
  rocks: [
    ['R1', 5, 1.5, 0.5, 0.8, 0.8],
    ['R2', 5, -2.5, 0.6, 0.5, 0.6],
    ['R3', 8, 0, 0.4, 1, 1],
    ['R4', 3, -4.5, 0.5, 0.5, 0.5],
    ['R5', 10.5, 5, 0.4, 0.4, 0.7],
  ],
  block: ['bloqueo', 9.7, 0, 1.1, 0.3, 0.8],
  sensor_height: 0.3,
  ray_angles: Array.from({ length: 9 }, (_, index) => -Math.PI / 2 + (index * Math.PI) / 8),
  actions: ['avanzar', 'avanzar + izq', 'avanzar + der', 'girar izq', 'girar der'],
  dt: 0.5,
  terrain: { n: 61, heights: buildTerrain(61) },
})

export const normalizeMap = (value: unknown): ExpeditionMap => {
  const record = asRecord(value)
  const fallback = createContractMap()
  const terrainRecord = asRecord(record.terrain)
  const terrainN = Math.max(2, asNumber(terrainRecord.n, fallback.terrain.n))
  const terrainHeights = numberArray(terrainRecord.heights)
  const expectedHeights = terrainN * terrainN

  const beacons = Array.isArray(record.beacons)
    ? record.beacons.map((item) => tuple(item, [0, 0] as [number, number]))
    : fallback.beacons

  const rocks = Array.isArray(record.rocks)
    ? record.rocks
        .filter((item): item is unknown[] => Array.isArray(item) && item.length >= 6)
        .map((item) => [
          asString(item[0], 'roca'),
          asNumber(item[1]),
          asNumber(item[2]),
          asNumber(item[3]),
          asNumber(item[4]),
          asNumber(item[5]),
        ] as ExpeditionMap['rocks'][number])
    : fallback.rocks

  const mission = asRecord(record.mission)
  const environment = asRecord(record.environment)
  const planet = asRecord(environment.planet)

  return {
    size: tuple(record.size, fallback.size),
    base: tuple(record.base, fallback.base),
    beacons,
    visit_radius: asNumber(record.visit_radius, fallback.visit_radius),
    crack: tuple(record.crack, fallback.crack),
    crack_depth: asNumber(record.crack_depth, fallback.crack_depth),
    sand: tuple(record.sand, fallback.sand),
    rocks,
    block: Array.isArray(record.block) && record.block.length >= 6
      ? [
          asString(record.block[0], 'bloqueo'),
          asNumber(record.block[1]),
          asNumber(record.block[2]),
          asNumber(record.block[3]),
          asNumber(record.block[4]),
          asNumber(record.block[5]),
        ]
      : fallback.block,
    sensor_height: asNumber(record.sensor_height, fallback.sensor_height),
    ray_angles: numberArray(record.ray_angles).length ? numberArray(record.ray_angles) : fallback.ray_angles,
    actions: Array.isArray(record.actions) ? record.actions.map((item) => asString(item, 'acción')) : fallback.actions,
    dt: asNumber(record.dt, fallback.dt),
    terrain: {
      n: terrainHeights.length === expectedHeights ? terrainN : fallback.terrain.n,
      heights: terrainHeights.length === expectedHeights ? terrainHeights : fallback.terrain.heights,
    },
    environment_version: asString(record.environment_version) || undefined,
    scenario_id: asString(record.scenario_id) || undefined,
    difficulty: asString(record.difficulty) || undefined,
    difficulty_level: record.difficulty_level === undefined ? undefined : asNumber(record.difficulty_level),
    active_extent: Array.isArray(record.active_extent)
      ? tuple(record.active_extent, fallback.size)
      : undefined,
    samples: normalizeSamples(record.samples),
    mission: Object.keys(mission).length ? {
      target_type: asString(mission.target_type, 'desconocida'),
      requested_samples: asNumber(mission.requested_samples),
      capacity: asNumber(mission.capacity),
    } : undefined,
    environment: Object.keys(environment).length ? {
      planet: Object.keys(planet).length ? {
        planet_id: asString(planet.planet_id) || undefined,
        gravity_m_s2: planet.gravity_m_s2 === undefined ? undefined : asNumber(planet.gravity_m_s2),
        atmosphere_density_kg_m3: planet.atmosphere_density_kg_m3 === undefined
          ? undefined
          : asNumber(planet.atmosphere_density_kg_m3),
      } : undefined,
      traction: environment.traction === undefined ? undefined : asNumber(environment.traction),
      wind_speed_m_s: environment.wind_speed_m_s === undefined ? undefined : asNumber(environment.wind_speed_m_s),
      wind_direction_rad: environment.wind_direction_rad === undefined ? undefined : asNumber(environment.wind_direction_rad),
      dust_opacity: environment.dust_opacity === undefined ? undefined : asNumber(environment.dust_opacity),
      temperature_c: environment.temperature_c === undefined ? undefined : asNumber(environment.temperature_c),
      visibility_m: environment.visibility_m === undefined ? undefined : asNumber(environment.visibility_m),
      dunes: Array.isArray(environment.dunes) ? environment.dunes.map(numberArray) : undefined,
      craters: Array.isArray(environment.craters) ? environment.craters.map(numberArray) : undefined,
      traction_zones: Array.isArray(environment.traction_zones)
        ? environment.traction_zones.map(numberArray)
        : undefined,
    } : undefined,
  }
}

const normalizeHistory = (value: unknown): HistoryPoint[] => {
  if (!Array.isArray(value)) return []
  return value.map((item, index) => {
    const record = asRecord(item)
    return {
      episode: asNumber(record.episode, index + 1),
      reward: asNumber(record.reward),
      success: asBoolean(record.success) ? 1 : asNumber(record.success),
      epsilon: record.epsilon === undefined ? undefined : asNumber(record.epsilon),
    }
  })
}

const normalizeRocks = (value: unknown): EpisodeRock[] | undefined => {
  if (!Array.isArray(value)) return undefined
  return value
    .filter((item): item is unknown[] => Array.isArray(item) && item.length >= 5)
    .map((item) => [asNumber(item[0]), asNumber(item[1]), asNumber(item[2]), asNumber(item[3]), asNumber(item[4])])
}

const extractEpisodeRecord = (value: unknown): UnknownRecord => {
  const root = asRecord(value)
  const candidates = [root.episode, root.replay, root.trajectory]
  for (const candidate of candidates) {
    if (isRecord(candidate)) return { ...root, ...candidate }
  }
  if (Array.isArray(root.episodes) && isRecord(root.episodes[0])) return { ...root, ...root.episodes[0] }
  return root
}

export const normalizeEpisode = (
  value: unknown,
  defaults: Partial<Pick<EpisodeData, 'mode' | 'runId' | 'modelName' | 'environmentVersion' | 'levelId'>> = {},
): EpisodeData => {
  const record = extractEpisodeRecord(value)
  const framesValue = Array.isArray(record.frames) ? record.frames : []
  const frames = framesValue.map(normalizeFrame).filter((frame): frame is ReplayFrame => Boolean(frame))
  if (!frames.length) throw new Error('La respuesta no contiene cuadros compatibles con el contrato del replay.')

  const metrics = asRecord(record.metrics)
  const results = Array.isArray(record.results) && isRecord(record.results[0]) ? record.results[0] : {}
  const merged = { ...metrics, ...results, ...record }
  const finalFrame = frames.at(-1) as ReplayFrame
  const rewardFromFrames = frames.reduce((total, frame) => total + frame[10], 0)

  return {
    episode: typeof merged.episode === 'string' || typeof merged.episode === 'number' ? merged.episode : 1,
    mode: asString(merged.mode, defaults.mode ?? 'evaluación'),
    runId: asString(merged.run_id, defaults.runId ?? 'sin-run'),
    modelName: asString(merged.model_name, asString(merged.algorithm, defaults.modelName ?? 'modelo')),
    environmentVersion: asString(merged.environment_version, defaults.environmentVersion ?? 'sin-versión'),
    levelId: asString(merged.level_id, defaults.levelId ?? 'n2'),
    frames,
    rocks: normalizeRocks(merged.rocks),
    block: asBoolean(merged.block),
    success: asBoolean(merged.success),
    end: asString(merged.end, asString(merged.end_reason, 'running')),
    reward: asNumber(merged.reward, rewardFromFrames),
    steps: asNumber(merged.steps, frames.length - 1),
    visited: asNumber(merged.visited, Math.max(0, finalFrame[7] - 1)),
    battery: asNumber(merged.battery, finalFrame[8]),
    pathLength: merged.path_length === undefined ? undefined : asNumber(merged.path_length),
    latencyMs: merged.latency_ms === undefined
      ? merged.mean_inference_latency_ms === undefined
        ? undefined
        : asNumber(merged.mean_inference_latency_ms)
      : asNumber(merged.latency_ms),
    history: normalizeHistory(merged.history ?? merged.history_tail),
    scenarioId: asString(merged.scenario_id) || undefined,
    difficulty: asString(merged.difficulty) || undefined,
    planet: asString(merged.planet) || undefined,
    objective: asString(merged.objective) || undefined,
    requestedSamples: merged.requested_samples === undefined ? undefined : asNumber(merged.requested_samples),
    samplesCollected: merged.samples_collected === undefined ? undefined : asNumber(merged.samples_collected),
    referencePath: Array.isArray(merged.reference_path) ? merged.reference_path as number[][] : [],
    pathEfficiency: typeof merged.path_efficiency === 'number' ? merged.path_efficiency : null,
    missionTimeline: normalizeMissionTimeline(merged.mission_timeline),
    actionsTimeline: Array.isArray(merged.actions_timeline)
      ? merged.actions_timeline.map((action) => asString(action)).filter(Boolean)
      : undefined,
  }
}

export const normalizeCatalog = (value: unknown): CatalogData => {
  const root = asRecord(value)
  const nested = isRecord(root.catalog) ? root.catalog : root
  const rawRuns = Array.isArray(nested.runs)
    ? nested.runs
    : Array.isArray(nested.models)
      ? nested.models
      : []

  const runs: RunOption[] = rawRuns.map((item, index) => {
    const run = asRecord(item)
    const metrics = asRecord(run.metrics)
    return {
      runId: asString(run.run_id, `run-${index + 1}`),
      modelName: asString(run.model_name, asString(run.algorithm, 'modelo')),
      levelId: asString(run.level_id, 'n2'),
      environmentVersion: asString(run.environment_version, 'sin-versión'),
      algorithm: asString(run.algorithm, 'modelo'),
      status: asString(run.status, 'unknown'),
      successRate: metrics.success_rate === undefined ? undefined : asNumber(metrics.success_rate),
    }
  })

  const recommended = nested.recommended_run_id
  const recommendedRunId = typeof recommended === 'string'
    ? recommended
    : isRecord(recommended)
      ? Object.values(recommended).find((item): item is string => typeof item === 'string')
      : undefined

  const recommendedByLevel: Record<string, string> = {}
  if (isRecord(recommended)) {
    for (const [level, runId] of Object.entries(recommended)) {
      if (typeof runId === 'string') recommendedByLevel[level] = runId
    }
  }

  return {
    runs,
    recommendedRunId,
    recommendedByLevel,
    servedVersion: asString(nested.served_version) || undefined,
  }
}

export const normalizeReplayBundle = (value: unknown): ReplayBundle => {
  const root = asRecord(value)
  const rawEpisodes = Array.isArray(root.episodes) && root.episodes.length
    ? root.episodes
    : isRecord(root.episode)
      ? [root.episode]
      : []
  if (!rawEpisodes.length) {
    throw new Error('El archivo debe incluir episode o una lista no vacía en episodes.')
  }
  const history = normalizeHistory(root.history)
  const map = normalizeMap(root.map)
  const episodes = rawEpisodes.map((episode) => {
    const normalized = normalizeEpisode(episode, {
      mode: 'entrenamiento-replay',
      runId: 'archivo-local',
      modelName: 'replay',
      environmentVersion: asString(root.environment_version, map.environment_version ?? 'archivo'),
      levelId: asString(root.level_id, map.difficulty ?? 'n2'),
    })
    return { ...normalized, mode: 'entrenamiento-replay', history: normalized.history.length ? normalized.history : history }
  })
  return { map, episodes, history }
}
