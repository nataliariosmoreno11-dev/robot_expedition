import type { EpisodeData, ExpeditionMap, FrameSnapshot } from '../types'
import { END_LABELS, FRAME_INDEX } from '../types'

const required = <T extends Element>(selector: string): T => {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Falta el elemento ${selector}.`)
  return element
}

const formatNumber = (value: number, digits = 1) =>
  new Intl.NumberFormat('es-CO', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value)

const modeLabel = (mode: string) => {
  const labels: Record<string, string> = {
    manual: 'MANUAL',
    random: 'ALEATORIO',
    aleatorio: 'ALEATORIO',
    rule: 'REGLA',
    regla: 'REGLA',
    evaluation: 'EVALUACIÓN',
    evaluación: 'EVALUACIÓN',
    entrenamiento: 'ENTRENAMIENTO · REPLAY',
    'entrenamiento-replay': 'ENTRENAMIENTO · REPLAY',
  }
  return labels[mode.toLowerCase()] ?? mode.toUpperCase()
}

export class MissionHud {
  private episode?: EpisodeData
  private modeBadge = required<HTMLElement>('#modeBadge')
  private sourceBadge = required<HTMLElement>('#sourceBadge')
  private apiStatus = required<HTMLElement>('#apiStatus')
  private notice = required<HTMLElement>('#notice')
  private noticeTitle = required<HTMLElement>('#noticeTitle')
  private noticeText = required<HTMLElement>('#noticeText')

  constructor() {
    required<HTMLButtonElement>('#dismissNotice').addEventListener('click', () => {
      this.notice.hidden = true
    })
  }

  setConnection(state: 'offline' | 'loading' | 'online', label: string) {
    this.apiStatus.className = `connection connection-${state}`
    this.apiStatus.innerHTML = '<i aria-hidden="true"></i>'
    this.apiStatus.append(document.createTextNode(label))
  }

  showNotice(title: string, text: string) {
    this.noticeTitle.textContent = title
    this.noticeText.textContent = text
    this.notice.hidden = false
  }

  hideNotice() {
    this.notice.hidden = true
  }

  setSource(source: 'api' | 'replay' | 'none', label?: string) {
    this.sourceBadge.classList.toggle('replay', source === 'replay')
    this.sourceBadge.textContent = label ?? (source === 'api' ? 'API' : source === 'replay' ? 'REPLAY' : 'SIN FUENTE')
    this.modeBadge.classList.toggle('replay', source === 'replay')
  }

  setEpisode(episode: EpisodeData, source: 'api' | 'replay', map?: ExpeditionMap) {
    this.episode = episode
    this.setSource(source, source === 'replay' ? 'REPLAY · ARCHIVO' : 'API · EVALUACIÓN')
    this.modeBadge.textContent = source === 'replay' ? `REPLAY · ${modeLabel(episode.mode)}` : modeLabel(episode.mode)
    required<HTMLElement>('#episodeValue').textContent = String(episode.episode)
    required<HTMLElement>('#runValue').textContent = episode.runId
    required<HTMLElement>('#modelValue').textContent = episode.modelName
    required<HTMLElement>('#environmentValue').textContent = episode.environmentVersion
    required<HTMLElement>('#levelValue').textContent = episode.levelId.toUpperCase()
    required<HTMLElement>('#successValue').textContent = episode.success ? 'Sí' : 'No'
    required<HTMLElement>('#successValue').style.color = episode.success ? 'var(--green)' : 'var(--red)'
    required<HTMLElement>('#pathValue').textContent = episode.pathLength === undefined ? '—' : `${formatNumber(episode.pathLength)} m`
    required<HTMLElement>('#visitedValue').textContent = episode.requestedSamples === undefined
      ? `${episode.visited}`
      : `${episode.samplesCollected ?? 0} / ${episode.requestedSamples}`
    required<HTMLElement>('#latencyValue').textContent = episode.latencyMs === undefined ? '—' : `${formatNumber(episode.latencyMs, 2)} ms`
    required<HTMLElement>('#planetValue').textContent = (episode.planet ?? map?.environment?.planet?.planet_id ?? '—').toUpperCase()
    required<HTMLElement>('#scenarioValue').textContent = episode.scenarioId ?? map?.scenario_id ?? '—'
    required<HTMLElement>('#missionValue').textContent = episode.objective
      ?? (map?.mission ? `Recolectar ${map.mission.requested_samples} × ${map.mission.target_type}` : 'Balizas y regreso')
    const environment = map?.environment
    required<HTMLElement>('#conditionsValue').textContent = environment
      ? `${formatNumber(environment.temperature_c ?? 0, 0)} °C · viento ${formatNumber(environment.wind_speed_m_s ?? 0)} m/s · polvo ${formatNumber((environment.dust_opacity ?? 0) * 100, 0)}%`
      : '—'
  }

  updateFrame(snapshot: FrameSnapshot) {
    if (!this.episode) return
    const { frame } = snapshot
    required<HTMLElement>('#objectiveValue').textContent = this.episode.requestedSamples !== undefined
      ? snapshot.objective
      : snapshot.objective === 'Base'
        ? 'Regresar a la base'
        : `Alcanzar la baliza ${snapshot.objective}`
    required<HTMLElement>('#stepValue').textContent = `${snapshot.index} / ${snapshot.total}`
    required<HTMLElement>('#rewardValue').textContent = formatNumber(snapshot.cumulativeReward, 2)
    required<HTMLElement>('#batteryValue').textContent = `${formatNumber(frame[FRAME_INDEX.battery])}%`
    required<HTMLElement>('#actionValue').textContent = snapshot.actionName
    required<HTMLElement>('#collisionsValue').textContent = String(snapshot.collisions)
    const mission = this.episode.missionTimeline?.[Math.min(snapshot.index, this.episode.missionTimeline.length - 1)]
    required<HTMLElement>('#eventValue').textContent = mission?.event ?? '—'
    required<HTMLElement>('#endValue').textContent = snapshot.index === snapshot.total
      ? (END_LABELS[this.episode.end] ?? this.episode.end)
      : 'En curso'

    const battery = Math.max(0, Math.min(100, frame[FRAME_INDEX.battery]))
    const bar = required<HTMLElement>('#batteryBar')
    bar.style.width = `${battery}%`
    bar.style.background = battery > 40 ? 'var(--green)' : battery > 15 ? 'var(--gold)' : 'var(--red)'

    required<HTMLInputElement>('#timeline').value = String(snapshot.index)
    required<HTMLElement>('#timeValue').textContent = `${this.formatTime(snapshot.index)} / ${this.formatTime(snapshot.total)}`
  }

  reset() {
    this.episode = undefined
    this.modeBadge.textContent = 'SIN DATOS'
    this.modeBadge.classList.remove('replay')
    this.setSource('none')
  }

  private formatTime(frame: number) {
    const seconds = Math.round(frame * 0.5)
    const minutesPart = String(Math.floor(seconds / 60)).padStart(2, '0')
    const secondsPart = String(seconds % 60).padStart(2, '0')
    return `${minutesPart}:${secondsPart}`
  }
}

export { required }
