import type { EpisodeSummary } from '../types'

interface ChartSeries {
  title: string
  color: string
  values: number[]
  suffix?: string
}

const movingMean = (values: number[], window = 10) => values.map((_, index) => {
  const chunk = values.slice(Math.max(0, index - window + 1), index + 1)
  return chunk.reduce((total, value) => total + value, 0) / Math.max(1, chunk.length)
})

export class TrainingDashboard {
  private context: CanvasRenderingContext2D
  private episodes: EpisodeSummary[] = []
  private canvas: HTMLCanvasElement

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Canvas 2D no disponible para el dashboard.')
    this.context = context
    new ResizeObserver(() => this.draw()).observe(canvas)
  }

  setEpisodes(episodes: EpisodeSummary[]) {
    this.episodes = [...episodes].sort((a, b) => a.episode_number - b.episode_number)
    this.draw()
  }

  private draw() {
    const ratio = Math.min(window.devicePixelRatio, 2)
    const width = Math.max(320, this.canvas.clientWidth)
    const height = Math.max(640, this.canvas.clientHeight)
    this.canvas.width = Math.floor(width * ratio)
    this.canvas.height = Math.floor(height * ratio)
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0)
    this.context.clearRect(0, 0, width, height)

    if (!this.episodes.length) {
      this.context.fillStyle = '#9f958b'
      this.context.font = '12px DM Mono, monospace'
      this.context.fillText('Sin episodios registrados.', 18, 30)
      return
    }

    const reward = this.episodes.map((episode) => episode.reward)
    const success = movingMean(this.episodes.map((episode) => episode.success ? 100 : 0))
    const charts: ChartSeries[] = [
      { title: 'Recompensa por episodio', color: '#f27d42', values: reward },
      { title: 'Media móvil de recompensa (10)', color: '#e6c36a', values: movingMean(reward) },
      { title: 'Tasa de éxito móvil (10)', color: '#7fd692', values: success, suffix: '%' },
      { title: 'Muestras recolectadas', color: '#61d5d2', values: this.episodes.map((episode) => episode.samples_collected) },
      { title: 'Colisiones por episodio', color: '#ff7164', values: this.episodes.map((episode) => episode.collisions) },
      { title: 'Dificultad curricular', color: '#c69cff', values: this.episodes.map((episode) => episode.difficulty_level) },
      { title: 'Distancia recorrida', color: '#83b5ff', values: this.episodes.map((episode) => episode.path_length), suffix: ' m' },
      { title: 'Duración', color: '#d9a17a', values: this.episodes.map((episode) => episode.duration_s), suffix: ' s' },
    ]
    charts.push(
      {title:'Energía consumida',color:'#ffb855',values:this.episodes.map(e=>e.energy_consumed ?? 0)},
      {title:'Muestras completadas',color:'#61d5d2',values:this.episodes.map(e=>100*e.samples_collected/Math.max(1,e.requested_samples)),suffix:'%'},
    )
    const comparable=this.episodes.filter(e=>e.success && e.path_efficiency!=null)
    if (comparable.length) charts.push({title:'Eficiencia de ruta ? ?xitos',color:'#c69cff',values:comparable.map(e=>e.path_efficiency!)})
    const columns = width >= 720 ? 2 : 1
    const rows = Math.ceil(charts.length / columns)
    const gap = 16
    const panelWidth = (width - gap * (columns - 1)) / columns
    const panelHeight = (height - gap * (rows - 1)) / rows
    charts.forEach((chart, index) => {
      const column = index % columns
      const row = Math.floor(index / columns)
      this.drawChart(chart, column * (panelWidth + gap), row * (panelHeight + gap), panelWidth, panelHeight)
    })
  }

  private drawChart(series: ChartSeries, x: number, y: number, width: number, height: number) {
    const ctx = this.context
    const padding = { top: 30, right: 14, bottom: 22, left: 42 }
    ctx.fillStyle = '#12100e'
    ctx.strokeStyle = 'rgba(255,239,222,0.12)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.roundRect(x + 0.5, y + 0.5, width - 1, height - 1, 7)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = '#f4efe9'
    ctx.font = '500 10px DM Mono, monospace'
    ctx.fillText(series.title.toUpperCase(), x + 12, y + 18)

    const minRaw = Math.min(...series.values)
    const maxRaw = Math.max(...series.values)
    const spread = Math.max(1e-6, maxRaw - minRaw)
    const min = minRaw - spread * 0.08
    const max = maxRaw + spread * 0.08
    const left = x + padding.left
    const right = x + width - padding.right
    const top = y + padding.top
    const bottom = y + height - padding.bottom
    ctx.strokeStyle = 'rgba(255,239,222,0.08)'
    for (let line = 0; line <= 3; line += 1) {
      const lineY = top + ((bottom - top) * line) / 3
      ctx.beginPath(); ctx.moveTo(left, lineY); ctx.lineTo(right, lineY); ctx.stroke()
    }
    ctx.fillStyle = '#9f958b'
    ctx.font = '9px DM Mono, monospace'
    ctx.fillText(`${maxRaw.toFixed(1)}${series.suffix ?? ''}`, x + 5, top + 4)
    ctx.fillText(`${minRaw.toFixed(1)}${series.suffix ?? ''}`, x + 5, bottom)
    ctx.fillText(`#${this.episodes[0].episode_number}`, left, y + height - 7)
    const lastLabel = `#${this.episodes.at(-1)?.episode_number ?? ''}`
    ctx.fillText(lastLabel, right - ctx.measureText(lastLabel).width, y + height - 7)

    ctx.strokeStyle = series.color
    ctx.lineWidth = 2
    ctx.beginPath()
    series.values.forEach((value, index) => {
      const pointX = left + (index / Math.max(1, series.values.length - 1)) * (right - left)
      const pointY = bottom - ((value - min) / Math.max(1e-6, max - min)) * (bottom - top)
      if (index === 0) ctx.moveTo(pointX, pointY)
      else ctx.lineTo(pointX, pointY)
    })
    ctx.stroke()
  }
}
