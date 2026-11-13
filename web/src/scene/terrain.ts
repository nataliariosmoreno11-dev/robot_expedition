import * as THREE from 'three'
import type { EpisodeRock, ExpeditionMap, MissionState } from '../types'

interface BeaconVisual {
  head: THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardMaterial>
  ring: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>
}

interface SampleVisual {
  root: THREE.Group
  material: THREE.MeshStandardMaterial
  target: boolean
}

const material = (color: THREE.ColorRepresentation, extra: THREE.MeshStandardMaterialParameters = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.92, ...extra })

const box = (
  width: number,
  height: number,
  depth: number,
  meshMaterial: THREE.MeshStandardMaterial,
) => {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), meshMaterial)
  mesh.castShadow = true
  mesh.receiveShadow = true
  return mesh
}

const createGroundTexture = () => {
  const canvas = document.createElement('canvas')
  canvas.width = 1024
  canvas.height = 1024
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas 2D no disponible.')

  context.fillStyle = '#bd6d3f'
  context.fillRect(0, 0, 1024, 1024)
  let seed = 3_188_291
  const random = () => {
    seed = (seed * 48_271) % 2_147_483_647
    return seed / 2_147_483_647
  }

  for (let index = 0; index < 2600; index += 1) {
    const red = 86 + random() * 64
    const green = 35 + random() * 32
    context.fillStyle = `rgba(${red}, ${green}, 18, ${0.04 + random() * 0.14})`
    context.fillRect(random() * 1024, random() * 1024, 2 + random() * 5, 2 + random() * 5)
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  return texture
}

const createLabel = (text: string, color = '#ffffff', size = 0.6) => {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 128
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas 2D no disponible.')
  context.font = '600 58px Manrope, sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'middle'
  context.lineWidth = 10
  context.strokeStyle = 'rgba(0, 0, 0, 0.66)'
  context.strokeText(text, 128, 64)
  context.fillStyle = color
  context.fillText(text, 128, 64)
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthWrite: false }),
  )
  sprite.scale.set(size * 2, size, 1)
  return sprite
}

export class TerrainWorld {
  readonly group = new THREE.Group()
  readonly rocks = new THREE.Group()
  width = 12
  height = 12
  worldScale = 1
  private map?: ExpeditionMap
  private beacons: BeaconVisual[] = []
  private samples = new Map<string, SampleVisual>()
  private block?: THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardMaterial>
  private dust?: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>

  constructor(scene: THREE.Scene) {
    scene.add(this.group, this.rocks)
  }

  build(map: ExpeditionMap) {
    this.map = map
    this.width = map.size[0]
    this.height = map.size[1]
    this.worldScale = 1.35 + THREE.MathUtils.clamp(map.difficulty_level ?? 1, 1, 8) * 0.22
    this.group.clear()
    this.rocks.clear()
    this.beacons = []
    this.samples.clear()
    this.dust = undefined

    const outside = new THREE.Mesh(new THREE.PlaneGeometry(90, 90, 24, 24), material(0x7a3b25))
    outside.rotation.x = -Math.PI / 2
    outside.position.y = -0.035
    outside.receiveShadow = true
    this.group.add(outside, this.buildTerrain(map))

    const wallMaterial = material(0x5f5550, { metalness: 0.08 })
    const thickness = 0.15
    const wallHeight = 0.58
    const walls: Array<[number, number, number, number]> = [
      [this.width / 2, this.height / 2 + thickness / 2, this.width + 2 * thickness, thickness],
      [this.width / 2, -this.height / 2 - thickness / 2, this.width + 2 * thickness, thickness],
      [-thickness / 2, 0, thickness, this.height],
      [this.width + thickness / 2, 0, thickness, this.height],
    ]
    for (const [x, y, width, depth] of (map.mission ? [] : walls)) {
      const wall = box(width, wallHeight, depth, wallMaterial)
      wall.position.copy(this.position(x, y, 0.2))
      this.group.add(wall)
    }

    this.addGeology(map)
    this.addAtmosphere(map)

    if (map.active_extent) {
      const [activeX, activeHeight] = map.active_extent
      const halfY = activeHeight / 2
      const boundary = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints([
          this.position(0, -halfY, 0.055),
          this.position(activeX, -halfY, 0.055),
          this.position(activeX, halfY, 0.055),
          this.position(0, halfY, 0.055),
        ]),
        new THREE.LineBasicMaterial({ color: 0xf27d42, transparent: true, opacity: 0.72 }),
      )
      this.group.add(boundary)
    }

    for (const zone of map.environment?.traction_zones ?? []) {
      const [x, y, halfX, halfY, traction] = zone
      const patch = new THREE.Mesh(
        new THREE.PlaneGeometry(halfX * 2, halfY * 2),
        new THREE.MeshBasicMaterial({
          color: traction < 0.5 ? 0x49464a : 0x766158,
          transparent: true,
          opacity: 0.34,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      )
      patch.rotation.x = -Math.PI / 2
      patch.scale.set(this.worldScale, this.worldScale, 1)
      patch.position.copy(this.position(x, y, 0.035))
      this.group.add(patch)
    }

    const [baseX, baseY, halfX, halfY] = map.base
    const landingPad = box(
      halfX * 2,
      0.045,
      halfY * 2,
      material(0x286f3c, { emissive: 0x092914, emissiveIntensity: 0.6 }),
    )
    landingPad.position.copy(this.position(baseX, baseY, 0.022))
    this.group.add(landingPad)
    const baseLabel = createLabel('BASE', '#cbffda', 0.48)
    baseLabel.position.copy(this.position(baseX, baseY - halfY - 0.35, 0.3))
    this.group.add(baseLabel)

    this.beacons = (map.mission ? [] : map.beacons).map(([x, y], index) => {
      const beacon = new THREE.Group()
      beacon.position.copy(this.position(x, y))
      const post = box(0.1, 0.82, 0.1, material(0x232a2d))
      post.position.y = 0.41
      beacon.add(post)
      const head = box(
        0.32,
        0.32,
        0.32,
        material(0x61d5d2, { emissive: 0x0d4547, emissiveIntensity: 1.2 }),
      )
      head.position.y = 1
      beacon.add(head)
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(Math.max(0.05, map.visit_radius - 0.05), map.visit_radius, 48),
        new THREE.MeshBasicMaterial({ color: 0x61d5d2, transparent: true, opacity: 0.58, side: THREE.DoubleSide }),
      )
      ring.rotation.x = -Math.PI / 2
      ring.position.y = 0.018
      beacon.add(ring)
      const label = createLabel(`B${index + 1}`, '#ffffff', 0.5)
      label.position.y = 1.48
      beacon.add(label)
      this.group.add(beacon)
      return { head, ring }
    })

    for (const sample of map.samples ?? []) {
      const root = new THREE.Group()
      root.position.copy(this.position(sample.x, sample.y, 0.16))
      const color = sample.target ? 0x66e0cb : 0xd59459
      const sampleMaterial = material(color, {
        emissive: sample.target ? 0x0c4f46 : 0x492712,
        emissiveIntensity: sample.target ? 0.85 : 0.28,
      })
      const specimen = new THREE.Mesh(new THREE.IcosahedronGeometry(0.18, 1), sampleMaterial)
      specimen.scale.set(1, 0.72, 0.85)
      specimen.castShadow = true
      root.add(specimen)
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.3, 0.36, 32),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: sample.target ? 0.82 : 0.32, side: THREE.DoubleSide }),
      )
      ring.rotation.x = -Math.PI / 2
      ring.position.y = -0.13
      root.add(ring)
      const typeLabel: Record<string, string> = { sulfato: 'A', hematita: 'B', basalto: 'C' }
      const label = createLabel(`${typeLabel[sample.sample_type] ?? sample.sample_id} · ${sample.sample_type}`, sample.target ? '#bfffee' : '#e6b68c', 0.3)
      label.position.y = 0.55
      root.add(label)
      root.visible = !sample.collected
      this.group.add(root)
      this.samples.set(sample.sample_id, { root, material: sampleMaterial, target: sample.target })
    }

    const [, blockX, blockY, blockHalfX, blockHalfY, blockHeight] = map.block
    this.block = box(
      blockHalfX * 2,
      blockHeight,
      blockHalfY * 2,
      material(0xff5546, { transparent: true, opacity: 0.86, emissive: 0x5d110b }),
    )
    this.block.position.copy(this.position(blockX, blockY, blockHeight / 2))
    this.block.visible = false
    this.group.add(this.block)
    this.setRocks()
  }

  setRocks(episodeRocks?: EpisodeRock[]) {
    if (!this.map) return
    this.rocks.clear()
    const rockMaterial = material(0x4b3027, { roughness: 1 })
    const rocks = episodeRocks ?? this.map.rocks.map(([, x, y, halfX, halfY, height]) => [x, y, halfX, halfY, height])
    for (const [x, y, halfX, halfY, height] of rocks) {
      const rock = new THREE.Mesh(new THREE.DodecahedronGeometry(0.72, 1), rockMaterial)
      rock.scale.set(halfX * 1.35, (height + 0.08) * 0.85, halfY * 1.35)
      rock.castShadow = true
      rock.receiveShadow = true
      rock.position.copy(this.position(x, y, height / 2 - 0.04))
      rock.rotation.y = (x * 0.31 + y * 0.17) % 0.3
      this.rocks.add(rock)
    }
  }

  private addGeology(map: ExpeditionMap) {
    const rock = material(0x512e22, { roughness: 1 })
    const ridge = new THREE.Group()
    const seed = [...(map.scenario_id ?? 'mars')].reduce((value, char) => value + char.charCodeAt(0), 0)
    const level = THREE.MathUtils.clamp(map.difficulty_level ?? 1, 1, 8)
    const peakCount = 8 + level * 3
    const worldRadius = 10.5 + level * 1.3
    for (let index = 0; index < peakCount; index += 1) {
      const angle = (index / peakCount) * Math.PI * 2
      const radius = worldRadius + ((seed * (index + 7)) % 11) * 0.36
      const peakHeight = 2.2 + level * 0.38 + (index % 6) * 0.62
      const peak = new THREE.Mesh(new THREE.ConeGeometry(2.1 + level * 0.22 + (index % 4), peakHeight, 7), rock)
      peak.position.set(Math.cos(angle) * radius, peakHeight * 0.36, Math.sin(angle) * radius)
      peak.rotation.y = angle * 1.7
      peak.scale.z = 0.55 + (index % 3) * 0.18
      peak.castShadow = true
      peak.receiveShadow = true
      ridge.add(peak)
    }
    this.group.add(ridge)

    for (const [x, y, amplitude, sigma] of map.environment?.craters ?? []) {
      const rim = new THREE.Mesh(
        new THREE.TorusGeometry(Math.max(0.35, sigma), 0.12 + Math.abs(amplitude) * 0.16, 8, 40),
        material(0x633724, { roughness: 1 }),
      )
      rim.rotation.x = -Math.PI / 2
      rim.position.copy(this.position(x, y, 0.02))
      rim.scale.set(this.worldScale, this.worldScale * 0.7, 1)
      rim.castShadow = true
      this.group.add(rim)
    }
  }

  private addAtmosphere(map: ExpeditionMap) {
    const opacity = map.environment?.dust_opacity ?? 0
    if (opacity < 0.04) return
    const count = 180 + Math.round(opacity * 820)
    const positions = new Float32Array(count * 3)
    let seed = [...(map.scenario_id ?? 'dust')].reduce((value, char) => value * 31 + char.charCodeAt(0), 17) >>> 0
    const random = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 4294967296
    }
    const spanX = this.width * this.worldScale * 1.35
    const spanZ = this.height * this.worldScale * 1.35
    for (let index = 0; index < count; index += 1) {
      positions[index * 3] = (random() - 0.5) * spanX
      positions[index * 3 + 1] = 0.08 + random() * (1.4 + opacity * 4.5)
      positions[index * 3 + 2] = (random() - 0.5) * spanZ
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    this.dust = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        color: 0xe09a69,
        size: 0.035 + opacity * 0.075,
        transparent: true,
        opacity: 0.16 + opacity * 0.42,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    )
    this.dust.frustumCulled = false
    this.group.add(this.dust)
  }

  tickWeather(delta: number) {
    if (!this.dust || !this.map) return
    const speed = 0.25 + (this.map.environment?.wind_speed_m_s ?? 0) * 0.035
    this.dust.position.x += delta * speed
    const limit = this.width * this.worldScale * 0.28
    if (this.dust.position.x > limit) this.dust.position.x = -limit
  }

  updateMission(stage: number, blockActive: boolean) {
    this.beacons.forEach(({ head, ring }, index) => {
      const completed = stage > index + 1
      const target = stage === index + 1
      head.material.color.set(completed ? 0x7fd692 : 0x61d5d2)
      head.material.emissive.set(completed ? 0x1b5a30 : target ? 0x1f7777 : 0x0d4547)
      ring.material.color.set(completed ? 0x7fd692 : 0x61d5d2)
      ring.material.opacity = target ? 0.9 : 0.45
    })
    if (this.block) this.block.visible = blockActive && stage >= 2
  }

  updateSamples(mission?: MissionState) {
    const byId = new Map((mission?.samples ?? []).map((sample) => [sample.sample_id, sample]))
    const inventory = new Set(mission?.inventory ?? [])
    for (const [id, visual] of this.samples) {
      const state = byId.get(id)
      const collected = Boolean(state?.collected) || inventory.has(id)
      visual.root.visible = !collected
      visual.material.emissiveIntensity = visual.target ? 0.85 : 0.28
    }
  }

  position(x: number, y: number, elevation = 0) {
    return new THREE.Vector3((x - this.width / 2) * this.worldScale, elevation, -y * this.worldScale)
  }

  setShadows(enabled: boolean) {
    for (const root of [this.group, this.rocks]) {
      root.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.castShadow = enabled
          object.receiveShadow = true
        }
      })
    }
  }

  private buildTerrain(map: ExpeditionMap) {
    const { n, heights } = map.terrain
    const geometry = new THREE.PlaneGeometry(
      this.width * this.worldScale,
      this.height * this.worldScale,
      n - 1,
      n - 1,
    )
    const positions = geometry.attributes.position
    const colors = new Float32Array(positions.count * 3)
    const [sandX, sandY, halfX, halfY] = map.sand

    for (let row = 0; row < n; row += 1) {
      for (let column = 0; column < n; column += 1) {
        const vertex = row * n + column
        const elevation = heights[(n - 1 - row) * n + column] ?? 0
        positions.setZ(vertex, elevation)
        const x = (column / (n - 1)) * this.width
        const y = this.height / 2 - (row / (n - 1)) * this.height
        let color: [number, number, number]
        if (elevation < -0.05) color = [0.22, 0.11, 0.07]
        else if (Math.abs(x - sandX) <= halfX && Math.abs(y - sandY) <= halfY) color = [1, 0.83, 0.57]
        else {
          const light = 0.86 + Math.min(elevation, 0.3) * 0.9
          color = [1 * light, 0.82 * light, 0.67 * light]
        }
        colors.set(color, vertex * 3)
      }
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ map: createGroundTexture(), vertexColors: true, roughness: 1 }),
    )
    mesh.rotation.x = -Math.PI / 2
    mesh.receiveShadow = true
    mesh.castShadow = true
    return mesh
  }
}
