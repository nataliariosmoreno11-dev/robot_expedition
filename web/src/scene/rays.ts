import * as THREE from 'three'

export class SensorRays {
  readonly group = new THREE.Group()
  private lines: Array<THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>>
  private enabled = true

  constructor(scene: THREE.Scene) {
    this.lines = Array.from({ length: 9 }, () => {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(1, 0, 0),
      ])
      const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: 0x61d5d2, transparent: true, opacity: 0.78 }),
      )
      this.group.add(line)
      return line
    })
    this.group.visible = false
    scene.add(this.group)
  }

  update(origin: THREE.Vector3, heading: number, angles: number[], distances: number[]) {
    this.group.position.copy(origin)
    this.group.rotation.set(0, heading, 0)
    this.lines.forEach((line, index) => {
      const length = Number.isFinite(distances[index]) ? distances[index] : 0
      const angle = angles[index] ?? 0
      const positions = line.geometry.attributes.position
      positions.setXYZ(1, Math.cos(angle) * length, 0, -Math.sin(angle) * length)
      positions.needsUpdate = true
      line.material.color.set(length > 0 && length < 0.6 ? 0xff7164 : 0x61d5d2)
    })
    this.group.visible = this.enabled
  }

  setVisible(visible: boolean) {
    this.enabled = visible
    this.group.visible = visible
  }
}
