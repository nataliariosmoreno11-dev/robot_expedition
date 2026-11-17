type SavedJob = {
  run_id?: string; status: string
  checkpoint?: { total_timesteps: number }
  progress?: { total_timesteps?: number }
}

export function mountViewerControls(options: {
  baseUrl: () => string | undefined
  replay: (payload: unknown, label: string) => void
}) {
  const get = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
  const viewer = document.querySelector<HTMLElement>('.viewport-panel')!
  const fullscreen = get<HTMLButtonElement>('fullscreenButton')
  const model = get<HTMLSelectElement>('historyModel')
  const scenario = get<HTMLSelectElement>('historyScenario')
  const episode = get<HTMLSelectElement>('historyEpisode')
  const play = get<HTMLButtonElement>('historyPlay')
  const notice = get<HTMLElement>('historyStatus')
  let selection = 0
  let loadingModels = false
  let loadedModels = false
  for (let level = 1; level <= 10; level++) scenario.add(new Option(`Escenario ${level}`, `scenario_${String(level).padStart(2, '0')}`))
  const syncFullscreen = () => {
    const expanded = document.fullscreenElement === viewer || viewer.classList.contains('viewer-expanded')
    fullscreen.textContent = expanded ? 'Salir de pantalla completa' : 'Pantalla completa'
    fullscreen.setAttribute('aria-pressed', String(expanded))
  }
  fullscreen.onclick = async () => {
    try {
      if (document.fullscreenElement === viewer) await document.exitFullscreen()
      else if (viewer.classList.contains('viewer-expanded')) viewer.classList.remove('viewer-expanded')
      else if (viewer.requestFullscreen && document.fullscreenEnabled) await viewer.requestFullscreen()
      else viewer.classList.add('viewer-expanded')
    } catch {
      viewer.classList.add('viewer-expanded')
    }
    syncFullscreen()
  }
  document.addEventListener('fullscreenchange', syncFullscreen)
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') { viewer.classList.remove('viewer-expanded'); syncFullscreen() }
  })
  const request = async (path: string) => {
    if (!options.baseUrl()) throw new Error('Conecta la API para consultar las versiones guardadas.')
    const response = await fetch(`${options.baseUrl()}${path}`)
    if (!response.ok) throw new Error(`No se pudo consultar el historial (${response.status}).`)
    return response.json()
  }
  const refreshModels = async () => {
    if (loadingModels || !options.baseUrl()) return
    loadingModels = true
    try {
      const result = await request('/training/jobs') as { jobs: SavedJob[] }
      const previous = model.value
      const jobs = result.jobs.filter(job => job.run_id && job.checkpoint)
        .sort((a, b) => (b.checkpoint?.total_timesteps ?? 0) - (a.checkpoint?.total_timesteps ?? 0))
      model.replaceChildren(...jobs.map(job => new Option(
        `${(job.checkpoint?.total_timesteps ?? 0).toLocaleString('es-CO')} pasos · ${job.run_id!.slice(0, 8)} · ${job.status}`, job.run_id)))
      if (jobs.some(job => job.run_id === previous)) model.value = previous
      if (!jobs.length) model.add(new Option('Todavía no hay versiones guardadas', ''))
      loadedModels = true
    } catch (error) { notice.textContent = String(error) }
    finally { loadingModels = false }
  }
  const invalidate = () => {
    selection++
    episode.replaceChildren(new Option('Pulsa Buscar intentos', ''))
    play.disabled = true
  }
  model.onchange = invalidate
  scenario.onchange = invalidate
  episode.onchange = () => { selection++; play.disabled = !episode.value }
  get<HTMLButtonElement>('historyLoad').onclick = async () => {
    invalidate()
    const token = selection
    notice.textContent = 'Buscando intentos registrados…'
    try {
      if (!model.value) throw new Error('Selecciona una versión guardada.')
      const payload = await request(`/episodes?run_id=${encodeURIComponent(model.value)}&scenario_id=${scenario.value}&limit=100`)
      if (token !== selection) return
      const rows = payload.episodes.filter((row: { success: boolean | null }) => row.success !== null)
      episode.replaceChildren(...rows.map((row: { episode_number: number; success: boolean; steps: number; collisions: number; mode?: string }) =>
        new Option(`#${row.episode_number} · ${row.success ? 'Misión cumplida' : 'Sin completar'} · ${row.steps} pasos · ${row.collisions} choques`, String(row.episode_number))))
      play.disabled = !rows.length
      notice.textContent = rows.length ? `${rows.length} intentos recientes de esta versión y escenario. Selecciona uno para comparar su recorrido.` : 'Esta versión no tiene intentos guardados en este escenario. Prueba otro escenario o modelo.'
    } catch (error) { if (token === selection) notice.textContent = String(error) }
  }
  play.onclick = async () => {
    const token = ++selection
    const run = model.value
    const number = episode.value
    play.disabled = true
    try {
      const payload = await request(`/episodes/${number}/replay`)
      if (token !== selection) return
      document.dispatchEvent(new Event('historical-replay-selected'))
      options.replay(payload, `REPLAY · ${run.slice(0, 8)} · EPISODIO ${number}`)
      get('progressReplayStatus').textContent = `Historial: modelo ${run.slice(0, 8)}, episodio ${number}.`
      document.querySelectorAll('[data-progress-level]').forEach(button => button.setAttribute('aria-pressed', 'false'))
      notice.textContent = `Reproduciendo modelo ${run.slice(0, 8)}, episodio ${number}. Es un intento histórico de entrenamiento.`
    } catch (error) { if (token === selection) notice.textContent = String(error) }
    finally { if (token === selection) play.disabled = false }
  }
  document.querySelector('.viewer-history')!.addEventListener('toggle', () => { void refreshModels() })
  window.setInterval(() => { if (!loadedModels) void refreshModels() }, 3000)
  void refreshModels()
  void (async () => {
    try {
      const response = await fetch('/demo/battery-audit.json')
      if (!response.ok) throw new Error('No se pudo cargar la prueba de batería.')
      const report = await response.json() as { generated_at: string; scenarios: { scenario: number; success: boolean; remaining_battery: number }[] }
      for (const result of report.scenarios) {
        const row = document.createElement('tr')
        for (const value of [String(result.scenario), `${result.remaining_battery.toFixed(1)} %`, result.success ? 'Completado' : 'Sin completar']) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell)
        }
        get('batteryAuditRows').append(row)
      }
      get('batteryAuditStatus').textContent = `Prueba registrada: ${new Date(report.generated_at).toLocaleString('es-CO')}. Estos resultados corresponden al catálogo de mapas utilizado en esa fecha.`
    } catch (error) { get('batteryAuditStatus').textContent = String(error) }
  })()
}
