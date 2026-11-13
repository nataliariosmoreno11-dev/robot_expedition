"""Independent clearance-aware grid planner. Never called by the policy/reward.

Costs are metres weighted by traction and slope; optimality is on this grid,
not a claim of an optimal continuous MuJoCo trajectory.
"""

import heapq
import itertools
import math
from functools import lru_cache

import numpy as np

from robot_expedition.envs.beacon_expedition import (
    BASE,
    CRACK,
    MAP_H,
    MAP_W,
    ROBOT_RADIUS,
)

from .scenario import _circle_hits_rect


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        queue = [remaining.pop()]
        while queue:
            i, j = queue.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    candidate = (i + di, j + dj)
                    if candidate in remaining:
                        remaining.remove(candidate)
                        queue.append(candidate)
    return count


def terrain_density(scenario, x, y):
    h = scenario.terrain_heights
    row = min(h.shape[0] - 2, max(1, round((y + MAP_H / 2) / MAP_H * (h.shape[0] - 1))))
    col = min(h.shape[1] - 2, max(1, round(x / MAP_W * (h.shape[1] - 1))))
    dx = (h[row, col + 1] - h[row, col - 1]) / (2 * MAP_W / (h.shape[1] - 1))
    dy = (h[row + 1, col] - h[row - 1, col]) / (2 * MAP_H / (h.shape[0] - 1))
    traction = min(
        [
            scenario.traction,
            *[
                z[4]
                for z in scenario.traction_zones
                if abs(x - z[0]) <= z[2] and abs(y - z[1]) <= z[3]
            ],
        ]
    )
    return 1 + 2 * (1 - traction) + 3 * math.atan(math.hypot(dx, dy))


@lru_cache(maxsize=10)
def campaign_reference(scenario_id):
    from .campaign import load_scenario

    return ReferencePlanner(load_scenario(scenario_id)).analyze()


def comparable_reference(scenario, start, collection_positions, finish):
    """Minimum grid tour through the same interaction positions as the rover.

    This avoids penalizing an agent that collects at the edge of the allowed
    radius or finishes at the edge of the base. No comparison for failed runs.
    """
    planner = ReferencePlanner(scenario)
    points = [start, *collection_positions]
    searches = [planner.search(p) for p in points]
    matrix = [
        [d.get(planner.cell(p), float("inf")) for p in points] for d, _ in searches
    ]
    # Tour's return cost refers to the actual finish, not the starting pose.
    for i, (distances, _) in enumerate(searches):
        matrix[i][0] = distances.get(planner.cell(finish), float("inf"))
    if any(not math.isfinite(v) for row in matrix for v in row):
        return None
    cost, order = planner._tour(matrix)
    route = []
    for a, b in itertools.pairwise(order):
        destination = finish if b == 0 else points[b]
        u = planner.cell(destination)
        prev = searches[a][1]
        leg = [u]
        while u != planner.cell(points[a]):
            u = prev[u]
            leg.append(u)
        route.extend(planner.point(c) for c in reversed(leg))
    return {
        "reference_path_cost": cost,
        "reference_path": route,
        "comparison": "same actual interaction positions and endpoints; optimal grid tour, 0.15 m resolution",
    }


class ReferencePlanner:
    def __init__(self, scenario, resolution=0.15):
        self.scenario = scenario
        self.resolution = resolution
        self.nx, self.ny = round(MAP_W / resolution), round(MAP_H / resolution)
        self.free = np.zeros((self.nx, self.ny), dtype=bool)
        self.weights = np.ones_like(self.free, dtype=float)
        dy, dx = np.gradient(
            scenario.terrain_heights,
            MAP_H / (scenario.terrain_heights.shape[0] - 1),
            MAP_W / (scenario.terrain_heights.shape[1] - 1),
        )
        self.slope = np.arctan(np.hypot(dx, dy))
        for i in range(self.nx):
            for j in range(self.ny):
                x, y = self.point((i, j))
                traction, slope = self.terrain(x, y)
                clearance = ROBOT_RADIUS + 0.20
                self.free[i, j] = (
                    clearance <= x <= scenario.active_x_max - clearance
                    and abs(y) <= scenario.active_y_half - clearance
                    and not any(
                        _circle_hits_rect(x, y, clearance, r)
                        for r in [CRACK, *scenario.rocks]
                    )
                    and slope <= min(0.32, math.atan(traction) * 0.65)
                )
                self.weights[i, j] = 1 + 2 * (1 - traction) + 3 * slope

    def point(self, cell):
        return (
            (cell[0] + 0.5) * self.resolution,
            (cell[1] + 0.5) * self.resolution - MAP_H / 2,
        )

    def cell(self, point):
        return (
            min(self.nx - 1, max(0, int(point[0] / self.resolution))),
            min(self.ny - 1, max(0, int((point[1] + MAP_H / 2) / self.resolution))),
        )

    def terrain(self, x, y):
        s = self.scenario
        row = min(
            self.slope.shape[0] - 1,
            max(0, round((y + MAP_H / 2) / MAP_H * (self.slope.shape[0] - 1))),
        )
        col = min(
            self.slope.shape[1] - 1,
            max(0, round(x / MAP_W * (self.slope.shape[1] - 1))),
        )
        traction = min(
            [
                s.traction,
                *[
                    z[4]
                    for z in s.traction_zones
                    if abs(x - z[0]) <= z[2] and abs(y - z[1]) <= z[3]
                ],
            ]
        )
        return traction, float(self.slope[row, col])

    def segment_cost(self, a, b):
        distance = math.dist(a, b)
        points = np.linspace(a, b, max(2, math.ceil(distance / self.resolution) + 1))
        costs = [1 + 2 * (1 - t) + 3 * s for t, s in (self.terrain(*p) for p in points)]
        return distance * sum(costs) / len(costs)

    def search(self, start):
        start = self.cell(start)
        if not self.free[start]:
            raise ValueError(f"start/target has insufficient clearance: {start}")
        distances = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost > distances[u]:
                continue
            for di, dj in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                v = (u[0] + di, u[1] + dj)
                if (
                    not (0 <= v[0] < self.nx and 0 <= v[1] < self.ny)
                    or not self.free[v]
                ):
                    continue
                if (
                    di
                    and dj
                    and (
                        not self.free[u[0] + di, u[1]] or not self.free[u[0], u[1] + dj]
                    )
                ):
                    continue
                new = (
                    cost
                    + self.resolution
                    * math.hypot(di, dj)
                    * (self.weights[u] + self.weights[v])
                    / 2
                )
                if new < distances.get(v, float("inf")):
                    distances[v] = new
                    previous[v] = u
                    heapq.heappush(queue, (new, v))
        return distances, previous

    @staticmethod
    def _tour(matrix):
        n = len(matrix) - 1
        dp = {(1 << i, i): (matrix[0][i + 1], [i + 1]) for i in range(n)}
        for mask in range(1, 1 << n):
            for last in range(n):
                if (mask, last) not in dp:
                    continue
                cost, path = dp[mask, last]
                for nxt in range(n):
                    if mask & (1 << nxt):
                        continue
                    key = (mask | 1 << nxt, nxt)
                    candidate = cost + matrix[last + 1][nxt + 1]
                    if candidate < dp.get(key, (float("inf"), []))[0]:
                        dp[key] = (candidate, [*path, nxt + 1])
        return min(
            (cost + matrix[last + 1][0], [0, *path, 0])
            for (mask, last), (cost, path) in dp.items()
            if mask == (1 << n) - 1
        )

    def analyze(self):
        points = [
            self.scenario.start[:2],
            *[(s.x, s.y) for s in self.scenario.target_samples],
        ]
        searches = [self.search(point) for point in points]
        matrix = [
            [d.get(self.cell(p), float("inf")) for p in points] for d, _ in searches
        ]
        for i, (distances, _) in enumerate(searches):
            matrix[i][0] = distances.get(self.cell(BASE[:2]), float("inf"))
        if any(not math.isfinite(v) for row in matrix for v in row):
            raise ValueError("mission disconnected by obstacles or unsafe slopes")
        cost, order = self._tour(matrix)
        if not math.isfinite(cost):
            raise ValueError("mission disconnected by obstacles or unsafe slopes")
        route = []
        for a, b in itertools.pairwise(order):
            previous = searches[a][1]
            u = self.cell(BASE[:2] if b == 0 else points[b])
            leg = [u]
            while u != self.cell(points[a]):
                u = previous[u]
                leg.append(u)
            route.extend(self.point(c) for c in reversed(leg))
        length = sum(math.dist(a, b) for a, b in itertools.pairwise(route))
        direct = [[math.dist(a, b) for b in points] for a in points]
        for i, a in enumerate(points):
            direct[i][0] = math.dist(a, BASE[:2])
        euclidean, _ = self._tour(direct)
        # Thin the clearance grid to a corridor skeleton, then count connected
        # junction regions rather than counting every open-space grid vertex.
        sk = self.free.copy()
        for _ in range(max(self.free.shape)):
            changed = False
            for phase in (0, 1):
                p = np.pad(sk, 1)
                ns = [
                    p[:-2, 1:-1],
                    p[:-2, 2:],
                    p[1:-1, 2:],
                    p[2:, 2:],
                    p[2:, 1:-1],
                    p[2:, :-2],
                    p[1:-1, :-2],
                    p[:-2, :-2],
                ]
                count = sum(n.astype(int) for n in ns)
                transitions = sum(
                    (~ns[k] & ns[(k + 1) % 8]).astype(int) for k in range(8)
                )
                triplets = (
                    ((0, 2, 4), (2, 4, 6)) if phase == 0 else ((0, 2, 6), (0, 4, 6))
                )
                remove = sk & (count >= 2) & (count <= 6) & (transitions == 1)
                for a, b, c in triplets:
                    remove &= ~(ns[a] & ns[b] & ns[c])
                if remove.any():
                    sk[remove] = False
                    changed = True
            if not changed:
                break
        p = np.pad(sk, 1)
        degree = np.zeros_like(sk, dtype=int)
        for di in range(3):
            for dj in range(3):
                if (di, dj) != (1, 1):
                    degree += p[di : di + self.nx, dj : dj + self.ny]
        dead_ends = int(np.sum(sk & (degree == 1)))
        branches = _component_count(sk & (degree >= 3))
        energy_scale = self.scenario.energy_scale
        # Conservative calibrated kinematic estimate; physical rollout is a
        # separate validation. Includes turns and collection/return overhead.
        steps = length / 0.22 + len(order) * 12
        energy = (
            steps
            * 0.55
            * energy_scale
            * (1 + max(0, -40 - self.scenario.temperature_c) / 180)
        )
        if steps > self.scenario.max_steps or energy > self.scenario.battery_budget:
            raise ValueError(
                f"mission budget insufficient: {steps:.0f} steps, {energy:.1f} energy"
            )
        return {
            "scenario_id": self.scenario.scenario_id,
            "reference_path": route,
            "reference_path_cost": cost,
            "shortest_feasible_path_length": length,
            "route_detour_ratio": length / euclidean,
            "dead_end_count": dead_ends,
            "relevant_branch_count": branches,
            "terrain_risk_score": float(self.weights[self.free].mean() - 1),
            "multi_target_route_cost": cost,
            "estimated_steps": steps,
            "estimated_energy": energy,
            "target_order": [
                self.scenario.target_samples[i - 1].sample_id for i in order[1:-1]
            ],
            "method": "Dijkstra + exact subset tour on clearance grid; kinematic budget estimate",
            "resolution": self.resolution,
        }
