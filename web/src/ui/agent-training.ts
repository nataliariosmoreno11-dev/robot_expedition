type Acceleration = {
  phase: string; level?: number; updates?: number; max_updates?: number; note?: string
  evaluation?: { completed: number; total: number; successes: number; level?: number } | null
  stages?: { stage: number; examples: number; imitation_accuracy?: number }[]
}

type Job = {
  job_id: string; status: string; run_id?: string; budget: number; parent_job_id?: string
  progress: { total_timesteps?: number; updates?: number }
  checkpoint?: { total_timesteps: number }; error?: string; versioning_error?: string
  evaluation?: { success_rate?: number; success_rate_by_difficulty?: Record<string, number> }; dvc_pointer?: string
}

export function mountAgentTraining(options: {
  baseUrl: () => string | undefined
  level: () => number
  replay: (payload: unknown, label: string) => void
  frame: (payload: unknown, label: string) => void
}) {
  const panel = document.createElement('section')
  panel.id = 'agentTraining'
  panel.className = 'panel-section agent-training'
  panel.innerHTML = `
    <span class="eyebrow">Aprendizaje por refuerzo</span><h2>Agente automático · PPO</h2>
    <p id="agentCampaign" role="status" class="section-copy">Consultando modelo final…</p>
    <button id="agentFinal" class="primary-button" disabled>Ver modelo final validado</button>
    <button id="agentCampaignStop" class="secondary-button" hidden>Detener campaña automática y guardar</button>
    <p class="section-copy">La red decide entre 7 acciones usando sensores, batería e historial de 4 observaciones. Aprende al recolectar, regresar y explorar; recibe penalizaciones por choques y consumo. Puede practicar con demostraciones durante el entrenamiento; al evaluar decide solo con sus sensores. Cada escenario aprobado conserva su propia pol?tica.</p>
    <p class="section-copy">Los intentos de entrenamiento sin explorar ni recolectar se reinician según la configuración de su versión. Las evaluaciones conservan las reglas originales.</p>
    <label class="field"><span>Pasos adicionales de entrenamiento</span><input id="agentBudget" type="number" value="65536" min="128" max="2000000" step="1024"></label>
    <label class="field"><span>Versión guardada / ejecución</span><select id="agentVersions"><option value="">Nueva política desde cero</option></select></label>
    <button id="agentLatest" class="secondary-button">Usar mejor versión evaluada</button>
    <p id="agentVersionNotice" class="section-copy"></p>
    <div class="replay-actions"><button id="agentStart" class="primary-button">Entrenar desde cero</button><button id="agentResume" class="secondary-button" disabled>Continuar pesos</button><button id="agentStop" class="secondary-button" disabled>Detener y guardar</button></div>
    <label class="section-copy"><input id="agentFollow" type="checkbox"> Mostrar replays del entrenamiento automáticamente</label>
    <p id="agentPhase" role="status" class="section-copy"></p>
    <progress id="agentPhaseProgress" max="1" value="0" style="width:100%" hidden></progress>
    <ul id="agentStages" class="section-copy"></ul>
    <p id="agentStatus" role="status">Conectando con el entrenador…</p>
    <progress id="agentProgress" max="1" value="0" style="width:100%"></progress>
    <p class="section-copy">Los replays muestran intentos guardados del agente. Cada continuación crea una versión nueva y conserva la anterior. El currículo avanza con ≥85% de éxito en al menos 50 evaluaciones independientes.</p>
    <div class="replay-actions"><button id="agentEvaluate" class="secondary-button" disabled>Ver política guardada en este escenario</button><button id="agentPauseView" class="secondary-button">Detener visualización</button></div>
    <button id="agentLog" class="secondary-button">Ver registro del entrenamiento</button><pre id="agentLogText" style="white-space:pre-wrap;max-height:180px;overflow:auto" hidden></pre>`
  document.querySelector('#planetaryMission')!.before(panel)
  const el = <T extends HTMLElement>(id: string) => panel.querySelector<T>(`#${id}`)!
  let jobs: Job[] = []
  let busy = false
  let polling = false
  let lastEpisode = 0
  let viewToken = 0
  let observedJob = ''
  let followLatest = true
  let evaluating = false
  let releaseJob: string | undefined
  let previewToken = 0
  document.addEventListener('historical-replay-selected', () => {
    previewToken++; viewToken++; evaluating = false
    el<HTMLInputElement>('agentFollow').checked = false
  })
  const preview = async (level: number) => {
    const token = ++previewToken
    viewToken++
    evaluating = false
    el<HTMLInputElement>('agentFollow').checked = false
    const notice = document.querySelector<HTMLElement>('#progressReplayStatus')!
    notice.textContent = `Cargando escenario ${level}…`
    try {
      const response = await fetch(`/demo/progress/scenario-${level}.json`)
      if (!response.ok) throw new Error('No se pudo cargar el recorrido guardado')
      const payload = await response.json()
      if (token !== previewToken) return
      options.replay(payload, `PPO · ESCENARIO ${level} · RECORRIDO EXITOSO GUARDADO`)
      notice.textContent = `Escenario ${level} · episodio ${payload.episode.episode} · ${payload.episode.steps} pasos · ${payload.episode.samples_collected}/${payload.episode.requested_samples} muestras · ${payload.episode.collisions} colisiones. Versión ${payload.episode.run_id.slice(0, 8)}.`
      document.querySelectorAll<HTMLButtonElement>('[data-progress-level]').forEach(button => {
        button.setAttribute('aria-pressed', String(Number(button.dataset.progressLevel) === level))
      })
    } catch (error) { if (token === previewToken) notice.textContent = String(error) }
  }
  document.querySelectorAll<HTMLButtonElement>('[data-progress-level]').forEach(button => {
    button.onclick = () => { void preview(Number(button.dataset.progressLevel)) }
  })
  void preview(1)
  const active = () => jobs.find(j => ['starting', 'running', 'stopping'].includes(j.status))
  const selected = () => jobs.find(j => j.job_id === el<HTMLSelectElement>('agentVersions').value)
  const request = async (path: string, payload?: unknown) => {
    const base = options.baseUrl()
    if (!base) throw new Error('API todavía no conectada')
    const response = await fetch(`${base}${path}`, {
      method: payload === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }
  const status = (text: string) => { el('agentStatus').textContent = text }
  const refresh = async () => {
    if (polling || !options.baseUrl()) return
    polling = true
    try {
      const result = await request('/training/jobs') as { enabled: boolean; jobs: Job[] }
      jobs = result.jobs
      const campaign = await request('/training/campaign') as {
        status: string; mastered?: number[]; current_level?: number; error?: string; acceleration?: Acceleration | null
        progress?: { total_timesteps?: number }
        release?: { job_id: string; run_id: string }
      }
      const phase = campaign.acceleration
      const labels: Record<string, string> = {
        preparing: 'Preparando pr?ctica', demonstration: 'Generando demostraciones verificadas',
        imitation: 'Practicando fragmentos de la misi?n', diagnostic: 'Prueba corta desde la base',
        mastery: 'Validaci?n completa del escenario', retention: 'Comprobando escenarios anteriores',
        replay: 'Guardando recorrido aprobado', approved: 'Escenario aprobado', ppo: 'Afinando mediante intentos y recompensas',
        all_scenarios_passed: 'Escenarios aprobados; falta certificaci?n final',
      }
      el('agentPhase').textContent = phase
        ? `Escenario ${phase.level ?? campaign.current_level ?? 1} ? ${labels[phase.phase] ?? phase.phase}${phase.evaluation && ['diagnostic', 'mastery', 'retention'].includes(phase.phase)
            ? ` ? nivel evaluado ${phase.evaluation.level ?? phase.level}: ${phase.evaluation.completed}/${phase.evaluation.total} intentos, ${phase.evaluation.successes} ?xitos`
            : phase.phase === 'imitation' ? ` ? ${phase.updates ?? 0}/${phase.max_updates ?? 0} actualizaciones supervisadas` : ''}`
        : 'Entrenamiento PPO. Consultando la fase actual.'
      const phaseProgress = el<HTMLProgressElement>('agentPhaseProgress')
      phaseProgress.hidden = !phase || !['imitation', 'diagnostic', 'mastery', 'retention'].includes(phase.phase)
      phaseProgress.value = phase?.phase === 'imitation'
        ? (phase.updates ?? 0) / Math.max(1, phase.max_updates ?? 1)
        : (phase?.evaluation?.completed ?? 0) / Math.max(1, phase?.evaluation?.total ?? 1)
      el('agentStages').replaceChildren(...(phase?.stages ?? []).map(stage => {
        const row = document.createElement('li')
        row.textContent = `Tramo ${stage.stage + 1} ? ${stage.examples} ejemplos${stage.imitation_accuracy === undefined ? '' : ` ? ${(stage.imitation_accuracy * 100).toFixed(0)}% de coincidencia con la demostraci?n`}. No equivale a misi?n aprobada.`
        return row
      }))
      releaseJob = campaign.release?.job_id
      const automated = ['running', 'validating', 'stopping'].includes(campaign.status)
      const overview = document.querySelector<HTMLElement>('#trainingOverview')!
      overview.textContent = releaseJob
        ? 'Modelo final disponible: validado en los 10 escenarios. Abre «Ver avance y modelos» para reproducirlo.'
        : automated
          ? `Entrenamiento en segundo plano · ${(campaign.progress?.total_timesteps ?? 0).toLocaleString('es-CO')} pasos · ${campaign.mastered?.length ?? 0}/10 escenarios superados en el currículo · escenario actual: ${campaign.current_level ?? 1}. La reproducción está desactivada por defecto; no necesitas mantener esta página abierta.`
          : `Campaña: ${campaign.status}. Todavía no hay modelo final validado. Consulta el avance y los registros.`
      el('agentCampaign').textContent = releaseJob
        ? `Modelo final ${campaign.release!.run_id.slice(0, 10)} · validado en los 10 escenarios. Pesos y pasos registrados.`
        : `Todavía no hay modelo final. Campaña: ${campaign.status}. ${campaign.mastered?.length ?? 0}/10 escenarios superados durante el currículo. ${campaign.error ?? ''} El entrenamiento funciona en segundo plano; no necesitas mirar los intentos.`
      el<HTMLButtonElement>('agentFinal').disabled = !releaseJob
      el<HTMLButtonElement>('agentCampaignStop').hidden = !automated || !result.enabled
      const select = el<HTMLSelectElement>('agentVersions')
      const previous = select.value
      select.replaceChildren(new Option('Nueva política desde cero', ''))
      for (const job of jobs) select.add(new Option(`${job.run_id?.slice(0, 10) ?? job.job_id.slice(0, 10)} · ${job.status} · ${job.progress?.total_timesteps ?? 0} pasos`, job.job_id))
      const current = active()
      const valid = jobs.filter(j => j.checkpoint && j.status !== 'failed')
      const evaluated = valid.filter(j => j.status === 'completed' && j.evaluation)
      const latest = evaluated.sort((a,b) => (b.evaluation?.success_rate ?? 0) - (a.evaluation?.success_rate ?? 0)
        || (b.checkpoint?.total_timesteps ?? 0) - (a.checkpoint?.total_timesteps ?? 0))[0]
        ?? valid.sort((a,b) => (b.checkpoint?.total_timesteps ?? 0) - (a.checkpoint?.total_timesteps ?? 0))[0]
      select.value = followLatest ? (releaseJob || latest?.job_id || jobs[0]?.job_id || '') : previous
      const chosen = selected()
      const job = current || chosen
      const behind = chosen && latest && chosen.job_id !== latest.job_id
        && (chosen.checkpoint?.total_timesteps ?? 0) < (latest.checkpoint?.total_timesteps ?? 0)
      el('agentVersionNotice').textContent = behind
        ? `Estás viendo una versión anterior de ${chosen.progress?.total_timesteps ?? 0} pasos. La última tiene ${latest.checkpoint?.total_timesteps} pasos.`
        : `Versión seleccionada: ${chosen?.run_id?.slice(0, 10) ?? 'pendiente'}${followLatest ? ' · entrenamiento activo o mejor versión evaluada' : ''}`
      el<HTMLButtonElement>('agentStart').disabled = automated || busy || Boolean(current) || !result.enabled
      el<HTMLButtonElement>('agentResume').disabled = automated || busy || Boolean(current) || !chosen?.checkpoint || !result.enabled
      el<HTMLButtonElement>('agentStop').disabled = !current || busy
      el<HTMLButtonElement>('agentEvaluate').disabled = !chosen?.checkpoint || chosen.job_id === current?.job_id
      if (job && !busy && !evaluating) {
        const parent = jobs.find(j => j.job_id === job.parent_job_id)
        const additional = Math.max(0, (job.progress?.total_timesteps ?? 0) - (parent?.checkpoint?.total_timesteps ?? 0))
        el<HTMLProgressElement>('agentProgress').value = Math.min(1, job.budget > 0 ? additional / job.budget : 1)
        const scenarioRate = job.evaluation?.success_rate_by_difficulty?.[String(options.level())]
        status(`${job.status} · ${job.progress?.total_timesteps ?? 0} pasos acumulados · ${job.progress?.updates ?? 0} actualizaciones PPO. ${job.error ?? ''} ${job.versioning_error ?? ''} ${job.dvc_pointer ? 'Pesos versionados con DVC.' : ''} ${job.evaluation ? `Éxito global: ${((job.evaluation.success_rate ?? 0) * 100).toFixed(1)}%. ${scenarioRate === undefined ? '' : `Escenario ${options.level()}: ${(scenarioRate * 100).toFixed(0)}%.`}` : 'Sin evaluación final; no implica dominio.'}`)
      } else if (!job) status(result.enabled ? 'Listo para entrenar. La política inicial aún no sabe resolver las misiones.' : 'Entrenamiento deshabilitado en este servidor.')
      if (job?.run_id && el<HTMLInputElement>('agentFollow').checked && !evaluating) {
        if (observedJob !== job.job_id) { lastEpisode = 0; observedJob = job.job_id }
        const payload = await request(`/episodes?run_id=${encodeURIComponent(job.run_id)}&mode=entrenamiento&limit=20`)
        const episode = payload.episodes.find((item: { episode_number: number; success: boolean | null }) => item.success !== null && item.episode_number > lastEpisode)
        if (episode) {
          lastEpisode = episode.episode_number
          options.replay(await request(`/episodes/${lastEpisode}/replay`), `REPLAY PPO · intento ${lastEpisode} · ${job.run_id.slice(0, 8)}`)
        }
      }
    } catch (error) {
      status(String(error))
      document.querySelector<HTMLElement>('#trainingOverview')!.textContent = 'No se pudo actualizar el estado del entrenamiento. Comprueba la conexión con la API.'
    } finally { polling = false }
  }
  const train = async (resume: boolean) => {
    busy = true
    try {
      viewToken++
      evaluating = false
      followLatest = true
      const parent = selected()
      const budget = Number(el<HTMLInputElement>('agentBudget').value)
      if (!Number.isInteger(budget) || budget < 128 || budget > 2000000) throw new Error('Elige entre 128 y 2.000.000 pasos')
      if (resume && !parent) throw new Error('Selecciona una versión')
      status('Preparando entrenamiento y registro de versión…')
      const job = await request('/training/jobs', { budget, parent_job_id: resume ? parent!.job_id : null })
      el<HTMLSelectElement>('agentVersions').add(new Option(job.job_id, job.job_id))
      el<HTMLSelectElement>('agentVersions').value = job.job_id
      el<HTMLInputElement>('agentFollow').checked = true
    } catch (error) { status(String(error)) } finally { busy = false; void refresh() }
  }
  el('agentStart').onclick = () => { void train(false) }
  el('agentResume').onclick = () => { void train(true) }
  el('agentStop').onclick = async () => {
    try { if (active()) { await request(`/training/jobs/${active()!.job_id}/stop`, {}); status('Guardando al finalizar la actualización PPO actual…') } }
    catch (error) { status(String(error)) }
  }
  el('agentLog').onclick = async () => {
    try {
      const job = active() || selected()
      if (!job) return
      el('agentLogText').hidden = false
      el('agentLogText').textContent = (await request(`/training/jobs/${job.job_id}/log`)).text
    } catch (error) { status(String(error)) }
  }
  el('agentPauseView').onclick = () => { viewToken++; evaluating = false; el<HTMLInputElement>('agentFollow').checked = false }
  el('agentEvaluate').onclick = async () => {
    const job = selected()
    if (!job) return
    const token = ++viewToken
    evaluating = true
    el<HTMLInputElement>('agentFollow').checked = false
    try {
      const reset = await request('/sim/reset', { environment: 'planetary', difficulty_level: options.level(), seed: 12345, policy: `trained:${job.job_id}` })
      const frames = [reset.frame]
      while (token === viewToken) {
        const step = await request('/sim/step', { session_id: reset.session_id })
        frames.push(step.frame)
        options.frame({ ...reset, ...step, frames, map: reset.map, run_id: job.run_id, mode: 'evaluación', algorithm: 'PPO' }, `PPO · EVALUACIÓN · ${job.run_id?.slice(0, 8)}`)
        if (step.terminated || step.truncated) { status(`Evaluación terminada: ${step.end}. Pesos sin modificar.`); break }
        await new Promise(resolve => window.setTimeout(resolve, 150))
      }
    } catch (error) { status(String(error)) } finally { if (token === viewToken) evaluating = false }
  }
  el('agentVersions').onchange = () => { viewToken++; evaluating = false; followLatest = false; lastEpisode = 0; void refresh() }
  el('agentFinal').onclick = () => {
    if (!releaseJob) return
    el<HTMLSelectElement>('agentVersions').value = releaseJob
    followLatest = false
    el<HTMLButtonElement>('agentEvaluate').click()
  }
  el('agentCampaignStop').onclick = async () => {
    try { await request('/training/campaign/stop', {}); status('Deteniendo la campaña y guardando los pesos…') }
    catch (error) { status(String(error)) }
  }
  el('agentLatest').onclick = () => { viewToken++; evaluating = false; followLatest = true; lastEpisode = 0; void refresh() }
  window.setInterval(() => { void refresh() }, 5000)
  void refresh()
}
