export type ReplayFrame = [
  x: number,
  y: number,
  z: number,
  qw: number,
  qx: number,
  qy: number,
  qz: number,
  stage: number,
  battery: number,
  action: number,
  reward: number,
  collision: number,
  rays: number[],
]

export type RockSpec = [name: string, x: number, y: number, halfX: number, halfY: number, height: number]
export type EpisodeRock = [x: number, y: number, halfX: number, halfY: number, height: number]

export interface TerrainSpec {
  n: number
  heights: number[]
}

export interface SampleSpec {
  sample_id: string
  sample_type: string
  x: number
  y: number
  target: boolean
  collected?: boolean
}

export interface MissionSpec {
  target_type: string
  requested_samples: number
  capacity: number
}

export interface EnvironmentSpec {
  planet?: { planet_id?: string; gravity_m_s2?: number; atmosphere_density_kg_m3?: number }
  traction?: number
  wind_speed_m_s?: number
  wind_direction_rad?: number
  dust_opacity?: number
  temperature_c?: number
  visibility_m?: number
  dunes?: number[][]
  craters?: number[][]
  traction_zones?: number[][]
}

export interface MissionState {
  target_type?: string
  requested?: number
  collected?: number
  inventory?: string[]
  samples?: SampleSpec[]
  event?: string | null
  status?: string
}

export interface ExpeditionMap {
  size: [number, number]
  base: [number, number, number, number]
  beacons: Array<[number, number]>
  visit_radius: number
  crack: [number, number, number, number]
  crack_depth: number
  sand: [number, number, number, number]
  rocks: RockSpec[]
  block: [string, number, number, number, number, number]
  sensor_height: number
  ray_angles: number[]
  actions: string[]
  dt: number
  terrain: TerrainSpec
  environment_version?: string
  scenario_id?: string
  difficulty?: string
  difficulty_level?: number
  active_extent?: [number, number]
  samples?: SampleSpec[]
  mission?: MissionSpec
  environment?: EnvironmentSpec
}

export interface HistoryPoint {
  episode: number
  reward: number
  success: number
  epsilon?: number
}

export interface EpisodeSummary {
  episode_number: number
  scenario_id: string
  difficulty_level: number
  requested_samples: number
  samples_collected: number
  success: boolean | null
  reward: number
  collisions: number
  path_length: number
  duration_s: number
  end: string
}

export interface EpisodeData {
  episode: number | string
  mode: string
  runId: string
  modelName: string
  environmentVersion: string
  levelId: string
  frames: ReplayFrame[]
  rocks?: EpisodeRock[]
  block?: boolean
  success: boolean
  end: string
  reward: number
  steps: number
  visited: number
  battery: number
  pathLength?: number
  latencyMs?: number
  history: HistoryPoint[]
  scenarioId?: string
  difficulty?: string
  planet?: string
  objective?: string
  requestedSamples?: number
  samplesCollected?: number
  missionTimeline?: MissionState[]
  actionsTimeline?: string[]
}

export interface RunOption {
  runId: string
  modelName: string
  levelId: string
  environmentVersion: string
  algorithm: string
  status: string
  successRate?: number
}

export interface CatalogData {
  runs: RunOption[]
  recommendedRunId?: string
  recommendedByLevel?: Record<string, string>
  servedVersion?: string
}

export interface ReplayBundle {
  map: ExpeditionMap
  episodes: EpisodeData[]
  history: HistoryPoint[]
}

export interface FrameSnapshot {
  frame: ReplayFrame
  index: number
  total: number
  cumulativeReward: number
  collisions: number
  actionName: string
  objective: string
}

export const FRAME_INDEX = {
  stage: 7,
  battery: 8,
  action: 9,
  reward: 10,
  collision: 11,
  rays: 12,
} as const

export const STAGE_NAMES = ['B1', 'B2', 'Base'] as const

export const END_LABELS: Record<string, string> = {
  exito: 'Misión cumplida',
  success: 'Misión cumplida',
  grieta: 'Cayó en la grieta',
  volco: 'Rover volcado',
  bateria: 'Batería agotada',
  tiempo: 'Tiempo agotado',
  collision: 'Colisión',
  roca: 'Colisión con roca',
  fuera_area: 'Fuera del área operativa',
  reinicio: 'Episodio reiniciado',
  interrumpido: 'Episodio interrumpido',
  running: 'En curso',
}

export interface LevelMetrics {
  success_rate: number
  failures_by_cause: Record<string, number>
  beacons_visited_mean?: number
  steps_mean: number
  collisions_per_episode: number
  final_battery_success?: number | null
  route_efficiency?: number | null
  return_mean: number
  latency_ms_mean: number
  latency_ms_p95: number
  episodes: number
  suite_id: string
  partial: boolean
  results_path?: string
}

export interface CatalogRun {
  run_id: string
  experiment_id: string
  model_name: string
  algorithm: string
  level_id: string
  status: string
  diagnosis?: string | null
  hypothesis?: string | null
  change?: string | null
  reference?: string | null
  seed?: number | null
  budget_timesteps?: number | null
  effective_budget_timesteps?: number | null
  training_time_s?: number | null
  parent_run_id?: string | null
  environment_version: string
  environment_hash?: string | null
  git_commit?: string | null
  commit_url?: string | null
  config_path?: string | null
  model_sha256?: string | null
  model_uri?: string | null
  mlflow_run_id?: string | null
  mlflow_run_name?: string | null
  mlflow_url?: string | null
  metrics?: { success_rate?: number }
  metrics_by_level: Partial<Record<'n1' | 'n2' | 'n3', LevelMetrics>>
  curves?: { training_success_rate?: Array<[number, number]>; training_reward?: Array<[number, number]> }
}

export interface CatalogDecision {
  run_id: string | null
  decision: string
  reason: string
}

export interface RichCatalog {
  catalog_version?: number
  served_version?: string
  published_at?: string
  published_by?: string
  git_commit?: string
  content_sha256?: string
  source?: string
  environment_version?: string
  recommended_run_id: Partial<Record<string, string | null>>
  pins?: Record<string, { run_id: string; reason: string; from_run_id?: string | null; at?: string }>
  decisions?: Record<string, CatalogDecision[]>
  selection_criteria?: { path: string; version: number; fixed_at: string; sha256: string }
  runs: CatalogRun[]
}

export interface MonitorRoute {
  requests: number
  errors: number
  latency_ms_mean: number | null
  latency_ms_p95: number | null
}

export interface MonitorData {
  routes: Record<string, MonitorRoute>
  last_errors: Array<{ at: string; route: string; status: number; detail?: string | null }>
  sessions: number
  uptime_s: number
}
