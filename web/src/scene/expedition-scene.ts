import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { EpisodeData, ExpeditionMap, FrameSnapshot, ReplayFrame } from '../types'
import { FRAME_INDEX, STAGE_NAMES } from '../types'
import { SensorRays } from './rays'
import { RoverModel } from './rover'
import { TerrainWorld } from './terrain'

interface SceneCallbacks {
  onFrame: (snapshot: FrameSnapshot) => void
  onPlaybackChange: (playing: boolean) => void
}

const MAX_TRAIL = 1200
const STAGE_COLORS = [
  new THREE.Color(0xf5f0ea),
  new THREE.Color(0xe6c36a),
  new THREE.Color(0x7fd692),
]

export class ExpeditionScene {
  private container: HTMLElement
  private callbacks: SceneCallbacks
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(56, 1, 0.1, 240)
  private controls: OrbitControls
  private sun = new THREE.DirectionalLight(0xffc08b, 3.4)
  private terrain: TerrainWorld
  private rover: RoverModel
  private rays: SensorRays
  private trailGeometry = new THREE.BufferGeometry()
  private trail: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>
  private map?: ExpeditionMap
  private episode?: EpisodeData
  private cumulativeRewards: number[] = []
  private framePosition = 0
  private lastFrameIndex = -1
  private speed = 4
  private follow = false
  private playingState = false
  private clock = new THREE.Clock()

  constructor(container: HTMLElement, callbacks: SceneCallbacks) {
    this.container = container
    this.callbacks = callbacks
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.32
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.container.prepend(this.renderer.domElement)

    this.scene.background = new THREE.Color(0x7c3e28)
    this.scene.fog = new THREE.FogExp2(0x8a472e, 0.014)
    this.camera.position.set(-8.5, 3.8, 9.5)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.maxPolarAngle = Math.PI * 0.495
    this.controls.minDistance = 2.2
    this.controls.maxDistance = 34

    this.scene.add(new THREE.HemisphereLight(0xffd7b0, 0x2b1610, 1.55))
    this.sun.position.set(-14, 10, 7)
    this.sun.castShadow = true
    this.sun.shadow.mapSize.set(2048, 2048)
    Object.assign(this.sun.shadow.camera, { left: -11, right: 11, top: 11, bottom: -11, near: 1, far: 40 })
    this.scene.add(this.sun)

    this.terrain = new TerrainWorld(this.scene)
    this.rover = new RoverModel(this.scene)
    this.rays = new SensorRays(this.scene)

    this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_TRAIL * 3), 3))
    this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_TRAIL * 3), 3))
    this.trail = new THREE.Line(this.trailGeometry, new THREE.LineBasicMaterial({ vertexColors: true }))
    this.scene.add(this.trail)

    new ResizeObserver(() => this.resize()).observe(this.container)
    this.renderer.setAnimationLoop(() => this.animate())
  }

  setMap(map: ExpeditionMap) {
    this.map = map
    this.terrain.build(map)
    const dust = map.environment?.dust_opacity ?? 0
    this.scene.fog = new THREE.FogExp2(dust > 0.55 ? 0x9a5538 : 0x7c3e28, 0.012 + dust * 0.032)
    this.scene.background = new THREE.Color(dust > 0.55 ? 0x9a5538 : 0x713520)
    const level = THREE.MathUtils.clamp(map.difficulty_level ?? 1, 1, 8)
    const extentX = map.active_extent?.[0] ?? map.size[0]
    const extentY = map.active_extent?.[1] ?? map.size[1]
    const center = this.terrain.position(extentX * 0.52, 0, 0.18 + level * 0.035)
    const visualExtent = Math.max(extentX, extentY) * this.terrain.worldScale
    const horizontalDistance = visualExtent * 0.82
    const cameraHeight = 2.7 + visualExtent * 0.18
    this.controls.target.copy(center)
    this.camera.position.set(
      center.x - horizontalDistance * 0.72,
      cameraHeight,
      center.z + horizontalDistance * 0.72,
    )
    this.controls.minDistance = Math.max(2.2, horizontalDistance * 0.28)
    this.controls.maxDistance = horizontalDistance * 2.5
    this.controls.update()
  }

  setEpisode(episode: EpisodeData) {
    this.episode = episode
    this.framePosition = 0
    this.lastFrameIndex = -1
    let total = 0
    this.cumulativeRewards = episode.frames.map((frame) => {
      total += frame[FRAME_INDEX.reward]
      return total
    })
    this.terrain.setRocks(episode.rocks)
    this.trailGeometry.setDrawRange(0, 0)
    this.setPlaying(false)
    this.applyFrame()
  }

  seek(index: number) {
    if (!this.episode) return
    this.framePosition = THREE.MathUtils.clamp(index, 0, this.episode.frames.length - 1)
    this.lastFrameIndex = -1
    this.applyFrame()
  }

  togglePlayback() {
    if (!this.episode) return false
    if (!this.playingState && this.framePosition >= this.episode.frames.length - 1) this.seek(0)
    this.setPlaying(!this.playingState)
    return this.playingState
  }

  setPlaying(playing: boolean) {
    this.playingState = Boolean(this.episode) && playing
    this.callbacks.onPlaybackChange(this.playingState)
  }

  setSpeed(speed: number) {
    this.speed = Math.max(0.25, speed)
  }

  setFollow(follow: boolean) {
    this.follow = follow
  }

  setLightMode(lightMode: boolean) {
    this.renderer.shadowMap.enabled = !lightMode
    this.renderer.setPixelRatio(lightMode ? 1 : Math.min(window.devicePixelRatio, 2))
    this.sun.castShadow = !lightMode
    this.terrain.setShadows(!lightMode)
    this.rover.setShadows(!lightMode)
    this.resize()
  }

  get playing() {
    return this.playingState
  }

  private animate() {
    const delta = Math.min(this.clock.getDelta(), 0.1)
    if (this.episode && this.map) {
      if (this.playingState) {
        const finalIndex = this.episode.frames.length - 1
        this.framePosition = Math.min(finalIndex, this.framePosition + (delta * this.speed) / this.map.dt)
        if (this.framePosition >= finalIndex) this.setPlaying(false)
      }
      this.applyFrame()
      if (this.follow) {
        const target = this.rover.group.position.clone()
        this.camera.position.add(target.clone().sub(this.controls.target).multiplyScalar(0.08))
        this.controls.target.lerp(target, 0.08)
      }
    }
    this.rover.tick()
    this.terrain.tickWeather(delta)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  private applyFrame() {
    if (!this.episode || !this.map) return
    const frames = this.episode.frames
    const finalIndex = frames.length - 1
    const index = Math.min(finalIndex, Math.floor(this.framePosition))
    const mix = Math.min(1, this.framePosition - index)
    const current = frames[index]
    const next = frames[Math.min(finalIndex, index + 1)]
    this.rover.applyInterpolated(current, next, mix, (x, y, elevation) => this.terrain.position(x, y, elevation))

    const forward = this.rover.forward()
    const rayData = next[FRAME_INDEX.rays].length ? next[FRAME_INDEX.rays] : current[FRAME_INDEX.rays]
    this.rays.update(
      this.rover.sensorPosition(this.map.sensor_height),
      Math.atan2(-forward.z, forward.x),
      this.map.ray_angles,
      rayData,
    )
    this.updateTrail(index, current)
    this.terrain.updateMission(current[FRAME_INDEX.stage], Boolean(this.episode.block))
    const mission = this.episode.missionTimeline?.[Math.min(index, this.episode.missionTimeline.length - 1)]
    this.terrain.updateSamples(mission)

    if (index !== this.lastFrameIndex) {
      const previous = frames[Math.max(0, index - 1)]
      if (index > 0 && current[FRAME_INDEX.collision] > previous[FRAME_INDEX.collision]) this.rover.flashCollision()
      const collisions = frames.slice(0, index + 1).reduce((total, frame) => total + frame[FRAME_INDEX.collision], 0)
      const action = current[FRAME_INDEX.action] >= 0 ? current[FRAME_INDEX.action] : next[FRAME_INDEX.action]
      const stage = current[FRAME_INDEX.stage]
      const collected = mission?.collected ?? this.episode.samplesCollected ?? 0
      const requested = mission?.requested ?? this.episode.requestedSamples
      const objective = requested === undefined
        ? (stage <= this.map.beacons.length ? STAGE_NAMES[Math.max(0, stage - 1)] : 'Base')
        : collected >= requested
          ? 'Regresar a la base'
          : `Recolectar ${collected}/${requested} · ${mission?.target_type ?? this.map.mission?.target_type ?? 'muestras'}`
      this.callbacks.onFrame({
        frame: current,
        index,
        total: finalIndex,
        cumulativeReward: this.cumulativeRewards[index] ?? 0,
        collisions,
        actionName: action >= 0 ? (this.map.actions[action] ?? `acción ${action}`) : '—',
        objective,
      })
      this.lastFrameIndex = index
    }
  }

  private updateTrail(index: number, interpolatedFrame: ReplayFrame) {
    if (!this.episode) return
    const positions = this.trailGeometry.attributes.position
    const colors = this.trailGeometry.attributes.color
    const count = Math.min(MAX_TRAIL, index + 2)
    for (let trailIndex = 0; trailIndex < count; trailIndex += 1) {
      const frame = trailIndex <= index ? this.episode.frames[trailIndex] : interpolatedFrame
      const point = this.terrain.position(frame[0], frame[1], frame[2] + 0.045)
      const color = STAGE_COLORS[Math.max(0, Math.min(2, frame[FRAME_INDEX.stage] - 1))]
      positions.setXYZ(trailIndex, point.x, point.y, point.z)
      colors.setXYZ(trailIndex, color.r, color.g, color.b)
    }
    positions.needsUpdate = true
    colors.needsUpdate = true
    this.trailGeometry.setDrawRange(0, count)
  }

  private resize() {
    const width = Math.max(1, this.container.clientWidth)
    const height = Math.max(1, this.container.clientHeight)
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }
}
