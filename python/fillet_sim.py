from __future__ import annotations

from dataclasses import dataclass, field
import html
import argparse
import random
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


ViewMode = Literal["surface", "density", "viscosity", "stiffness", "lattice"]
DeviceLike = torch.device | str


@dataclass
class Params:
    weight: float = 285.0
    length: float = 182.0
    width: float = 96.0
    thickness: float = 31.0
    taper: float = 0.58
    asymmetry: float = 0.05
    curvature: float = 0.035
    irregularity: float = 0.035
    density: float = 1055.0
    heterogeneity: float = 0.18
    viscosity: float = 0.58
    stiffness: float = 31.0
    recovery: float = 0.42
    friction: float = 0.48
    target_weight: float = 160.0
    tolerance: float = 5.0
    cut_angle: float = 0.0
    knife_speed: float = 48.0
    blade_depth: float = 36.0
    finger_radius: float = 24.0
    finger_force: float = 0.62


@dataclass
class Poke:
    center: torch.Tensor
    radius: float
    depth: float
    push: torch.Tensor
    amplitude: float = 1.0
    held: bool = False


@dataclass
class Candidate:
    angle: float
    offset: float
    target_side: Literal["low", "high"]
    portion_a: float
    portion_b: float
    error: float
    path_length: float
    score: float
    confidence: float
    points: torch.Tensor


@dataclass
class SurfaceData:
    rest: torch.Tensor
    current: torch.Tensor
    density: torch.Tensor
    viscosity: torch.Tensor
    stiffness: torch.Tensor


@dataclass
class LatticeData:
    rest: torch.Tensor
    current: torch.Tensor
    density: torch.Tensor
    viscosity: torch.Tensor
    stiffness: torch.Tensor
    mass_raw: torch.Tensor
    mass: torch.Tensor


@dataclass
class SimulationState:
    params: Params = field(default_factory=Params)
    seed: int = 1
    surface: SurfaceData | None = None
    lattice: LatticeData | None = None
    pokes: list[Poke] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    selected_candidate: int = 0
    approved: bool = False
    cut_done: bool = False


class FilletSimulation:
    def __init__(self, params: Params | None = None, device: DeviceLike = "cpu") -> None:
        self.params = params or Params()
        self.device = torch.device(device)
        self.state = SimulationState(params=self.params)

    def to(self, device: DeviceLike) -> "FilletSimulation":
        self.device = torch.device(device)
        if self.state.surface is not None:
            self.state.surface = self._move_surface(self.state.surface)
        if self.state.lattice is not None:
            self.state.lattice = self._move_lattice(self.state.lattice)
        self.state.pokes = [self._move_poke(poke) for poke in self.state.pokes]
        for candidate in self.state.candidates:
            candidate.points = candidate.points.to(self.device)
        return self

    def rebuild(self) -> None:
        self.state.surface = self.create_surface_samples()
        self.state.lattice = self.create_lattice_samples()
        self.state.pokes.clear()
        self.state.candidates = self.optimize_paths()
        self.state.selected_candidate = 0
        self.state.approved = False
        self.state.cut_done = False

    def step(self, dt: float) -> None:
        self.update_poke_recovery(dt)
        self.update_geometry_positions()
        self.state.candidates = self.optimize_paths()

    def clamp01(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, 0.0, 1.0)

    def lerp(self, a: torch.Tensor | float, b: torch.Tensor | float, t: torch.Tensor | float) -> torch.Tensor:
        a_t = torch.as_tensor(a, device=self.device, dtype=torch.float32)
        b_t = torch.as_tensor(b, device=self.device, dtype=torch.float32)
        t_t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        return a_t + (b_t - a_t) * t_t

    def seeded_noise(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        seed_value = self.state.seed if seed is None else seed
        s1 = torch.sin(x * 0.071 + y * 0.109 + z * 0.047 + seed_value * 1.713)
        s2 = torch.sin(x * 0.023 - y * 0.057 + z * 0.083 + seed_value * 4.311)
        s3 = torch.cos(x * 0.137 + y * 0.031 - z * 0.059 + seed_value * 0.997)
        return s1 * 0.5 + s2 * 0.3 + s3 * 0.2

    def profile_at(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params
        u = self.clamp01((x + p.length / 2.0) / p.length)
        end_envelope = torch.pow(torch.clamp(torch.sin(torch.pi * (0.035 + 0.93 * u)), min=0.0), 0.62)
        tip_growth = self.lerp(0.46, 1.0, torch.pow(u, max(0.35, p.taper)))
        shoulder = 1.0 + 0.09 * torch.exp(-torch.pow((u - 0.72) / 0.22, 2))
        wave = 1.0 + p.irregularity * (
            0.55 * torch.sin(u * 17.3 + self.state.seed)
            + 0.25 * torch.sin(u * 31.7 + 0.8)
        )
        half_width = torch.clamp(p.width * 0.5 * end_envelope * tip_growth * shoulder * wave, min=0.8)
        thick_envelope = torch.pow(torch.clamp(torch.sin(torch.pi * (0.015 + 0.95 * u)), min=0.0), 0.78)
        thick_growth = self.lerp(0.55, 1.0, torch.pow(u, 0.62))
        thickness = torch.clamp(
            p.thickness * thick_envelope * thick_growth * (1.0 + p.irregularity * 0.18 * torch.sin(u * 21.0 + 1.7)),
            min=1.2,
        )
        center_y = p.curvature * p.width * torch.sin((u - 0.1) * torch.pi) + p.asymmetry * p.width * (u - 0.5) * 0.52
        return half_width, thickness, center_y, u

    def top_height_at(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        half_width, thickness, center_y, _ = self.profile_at(x)
        q = torch.abs(y - center_y) / torch.clamp(half_width, min=0.1)
        dome = torch.pow(torch.clamp(1.0 - q * q, min=0.0), 0.66)
        side_bias = 1.0 + self.params.asymmetry * 0.18 * ((y - center_y) / torch.clamp(half_width, min=1.0))
        base = torch.tensor(5.2, device=self.device)
        return torch.where(q >= 1.0, base, base + thickness * dome * side_bias)

    def local_material(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.params
        half_width, thickness, _, u = self.profile_at(x)
        thick_norm = self.clamp01((z - 5.2) / torch.clamp(thickness, min=1.0))
        n1 = self.seeded_noise(x, y, z, self.state.seed)
        n2 = self.seeded_noise(x * 1.4 + 13.0, y * 0.8 - 9.0, z * 1.1, self.state.seed + 7)
        n3 = self.seeded_noise(x * 0.65 - 21.0, y * 1.3 + 4.0, z * 1.6, self.state.seed + 13)
        density = p.density * (1.0 + p.heterogeneity * 0.11 * n1 + 0.018 * thick_norm + 0.014 * u)
        viscosity = self.clamp01(p.viscosity + p.heterogeneity * 0.32 * n2 + 0.06 * (1.0 - thick_norm))
        stiffness = torch.clamp(p.stiffness * (1.0 + p.heterogeneity * 0.48 * n3 + 0.12 * u), min=4.0)
        return density, viscosity, stiffness

    def apply_pokes(self, points: torch.Tensor) -> torch.Tensor:
        if not self.state.pokes:
            return points.clone()

        out = points.clone()
        z_norm = self.clamp01((out[:, 2] - 5.2) / torch.clamp(out[:, 2].amax() - 5.2, min=1.0))
        for poke in self.state.pokes:
            if poke.amplitude < 0.001:
                continue
            dx = out[:, 0] - poke.center[0]
            dy = out[:, 1] - poke.center[1]
            r2 = dx * dx + dy * dy
            sigma2 = max(poke.radius * poke.radius, 1e-6)
            w = torch.exp(-r2 / (2.0 * sigma2))
            r = torch.sqrt(r2)
            ring = torch.exp(-torch.pow((r - poke.radius * 0.72) / max(2.0, poke.radius * 0.34), 2))
            body = 0.28 + 0.72 * z_norm
            slip = 0.22 + 0.78 * (1.0 - self.params.friction)
            out[:, 0] += poke.push[0] * w * body * poke.amplitude * slip
            out[:, 1] += poke.push[1] * w * body * poke.amplitude * slip
            out[:, 2] -= poke.depth * w * body * poke.amplitude
            out[:, 2] += poke.depth * 0.19 * ring * body * poke.amplitude
            out[:, 1] += self.params.asymmetry * poke.depth * 0.06 * w * poke.amplitude
        out[:, 2] = torch.clamp(out[:, 2], min=5.0)
        return out

    def create_surface_samples(self) -> SurfaceData:
        p = self.params
        ring_count = 70
        ring_segments = 38
        xs = []
        ys = []
        zs = []
        densities = []
        viscosities = []
        stiffnesses = []

        for i in range(ring_count):
            u = i / max(1, ring_count - 1)
            x = -p.length / 2.0 + u * p.length
            x_t = torch.tensor(x, device=self.device)
            half_width, thickness, center_y, _ = self.profile_at(x_t)
            for j in range(ring_segments):
                theta = torch.tensor((j / ring_segments) * 2.0 * torch.pi, device=self.device)
                sin_t = torch.sin(theta)
                cos_t = torch.cos(theta)
                lateral_asym = 1.0 + p.asymmetry * 0.14 * torch.sign(sin_t) * (0.3 + 0.7 * u)
                y = center_y + half_width * sin_t * lateral_asym
                vertical01 = torch.pow((cos_t + 1.0) * 0.5, 0.70)
                top_variation = 1.0 + p.irregularity * 0.08 * self.seeded_noise(x_t, y, vertical01 * thickness, self.state.seed + 22)
                z = 5.2 + thickness * vertical01 * top_variation
                xs.append(x_t)
                ys.append(y)
                zs.append(z)
                density, viscosity, stiffness = self.local_material(x_t, y, z)
                densities.append(density)
                viscosities.append(viscosity)
                stiffnesses.append(stiffness)

        rest = torch.stack([torch.stack(xs), torch.stack(ys), torch.stack(zs)], dim=1)
        current = rest.clone()
        return SurfaceData(
            rest=rest,
            current=current,
            density=torch.stack(densities).flatten(),
            viscosity=torch.stack(viscosities).flatten(),
            stiffness=torch.stack(stiffnesses).flatten(),
        )

    def create_lattice_samples(self) -> LatticeData:
        p = self.params
        nx, ny, nz = 27, 15, 9
        dx = p.length / max(1, nx - 1)
        points = []
        densities = []
        viscosities = []
        stiffnesses = []
        raw_mass = []

        for ix in range(nx):
            u = ix / max(1, nx - 1)
            x = -p.length / 2.0 + u * p.length
            x_t = torch.tensor(x, device=self.device)
            half_width, _, center_y, _ = self.profile_at(x_t)
            dy = (half_width * 2.0) / max(1, ny - 1)
            for iy in range(ny):
                y_norm = -1.0 + (iy / max(1, ny - 1)) * 2.0
                y = center_y + y_norm * half_width
                top = self.top_height_at(x_t, y)
                local_h = torch.clamp(top - 5.2, min=0.0)
                if float(local_h) < 0.6:
                    continue
                dz = local_h / max(1, nz - 1)
                for iz in range(nz):
                    z = 5.2 + (iz / max(1, nz - 1)) * local_h
                    if abs(y_norm) > 0.96 and iz > nz * 0.5:
                        continue
                    point = torch.tensor([x, float(y), float(z)], device=self.device)
                    density, viscosity, stiffness = self.local_material(point[0], point[1], point[2])
                    volume_mm3 = max(0.1, dx * max(0.1, float(dy)) * max(0.1, float(dz)))
                    points.append(point)
                    densities.append(density)
                    viscosities.append(viscosity)
                    stiffnesses.append(stiffness)
                    raw_mass.append(density * volume_mm3)

        rest = torch.stack(points, dim=0) if points else torch.zeros((0, 3), device=self.device)
        current = rest.clone()
        density = torch.stack(densities).flatten() if densities else torch.zeros((0,), device=self.device)
        viscosity = torch.stack(viscosities).flatten() if viscosities else torch.zeros((0,), device=self.device)
        stiffness = torch.stack(stiffnesses).flatten() if stiffnesses else torch.zeros((0,), device=self.device)
        mass_raw = torch.stack(raw_mass).flatten() if raw_mass else torch.zeros((0,), device=self.device)
        mass = mass_raw.clone()
        if mass.numel() > 0:
            mass *= p.weight / torch.clamp(mass_raw.sum(), min=1e-9)
        return LatticeData(
            rest=rest,
            current=current,
            density=density,
            viscosity=viscosity,
            stiffness=stiffness,
            mass_raw=mass_raw,
            mass=mass,
        )

    def update_geometry_positions(self) -> None:
        if self.state.surface is not None:
            self.state.surface.current = self.apply_pokes(self.state.surface.rest)
        if self.state.lattice is not None:
            self.state.lattice.current = self.apply_pokes(self.state.lattice.rest)

    def update_poke_recovery(self, dt: float) -> bool:
        changed = False
        updated: list[Poke] = []
        for poke in self.state.pokes:
            if not poke.held:
                rate = self.params.recovery * (1.2 - self.params.viscosity * 0.72)
                next_amplitude = poke.amplitude * float(torch.exp(torch.tensor(-dt * max(0.02, rate))))
                if abs(next_amplitude - poke.amplitude) > 0.00005:
                    changed = True
                poke.amplitude = next_amplitude
            if poke.amplitude > 0.012 or poke.held:
                updated.append(poke)
        if len(updated) != len(self.state.pokes):
            changed = True
        self.state.pokes = updated
        return changed

    def compute_path_points(self, angle: float, offset: float, samples: int = 34) -> torch.Tensor:
        if self.state.lattice is None or self.state.surface is None or self.state.lattice.current.numel() == 0:
            return torch.zeros((0, 3), device=self.device)

        angle_t = torch.tensor(angle, device=self.device)
        normal = torch.stack([torch.cos(angle_t), -torch.sin(angle_t)])
        tangent = torch.stack([torch.sin(angle_t), torch.cos(angle_t)])
        projected = self.state.lattice.current[:, :2] @ normal
        band = 5.5
        mask = torch.abs(projected - offset) < band
        if mask.any():
            tvals = self.state.lattice.current[mask, :2] @ tangent
            min_t = float(tvals.min()) - 2.0
            max_t = float(tvals.max()) + 2.0
        else:
            min_t = -self.params.width * 0.5
            max_t = self.params.width * 0.5

        ts = torch.linspace(min_t, max_t, samples, device=self.device)
        points = []
        surface_xy = self.state.surface.current[:, :2]
        surface_z = self.state.surface.current[:, 2]
        for t in ts:
            row = normal * offset + tangent * t
            deltas = surface_xy - row
            d2 = torch.sum(deltas * deltas, dim=1)
            index = int(torch.argmin(d2))
            points.append(torch.tensor([row[0], row[1], surface_z[index] + 1.9], device=self.device))
        return torch.stack(points, dim=0)

    def path_length(self, points: torch.Tensor) -> float:
        if points.numel() == 0 or points.shape[0] < 2:
            return 0.0
        return float(torch.sum(torch.linalg.norm(points[1:] - points[:-1], dim=1)))

    def optimize_paths(self) -> list[Candidate]:
        if self.state.lattice is None or self.state.lattice.mass.numel() == 0:
            return []

        total = float(self.state.lattice.mass.sum())
        target = min(self.params.target_weight, total - 20.0)
        preferred = torch.deg2rad(torch.tensor(self.params.cut_angle, device=self.device))
        candidates: list[Candidate] = []

        for deg_delta in range(-24, 25, 4):
            angle = float(preferred + torch.deg2rad(torch.tensor(float(deg_delta), device=self.device)))
            angle_t = torch.tensor(angle, device=self.device)
            normal = torch.stack([torch.cos(angle_t), -torch.sin(angle_t)])
            projected = self.state.lattice.current[:, :2] @ normal
            order = torch.argsort(projected)
            cumulative = 0.0
            best_low = {"error": float("inf"), "offset": 0.0, "mass": 0.0}
            best_high = {"error": float("inf"), "offset": 0.0, "mass": 0.0}
            for idx in order.tolist():
                cumulative += float(self.state.lattice.mass[idx])
                p = float(projected[idx])
                low_err = abs(cumulative - target)
                if low_err < best_low["error"]:
                    best_low = {"error": low_err, "offset": p, "mass": cumulative}
                high_mass = total - cumulative
                high_err = abs(high_mass - target)
                if high_err < best_high["error"]:
                    best_high = {"error": high_err, "offset": p, "mass": high_mass}

            for side, best in (("low", best_low), ("high", best_high)):
                points = self.compute_path_points(angle, best["offset"])
                length = self.path_length(points)
                angle_penalty = abs(deg_delta) * 0.022
                length_penalty = max(0.0, length - self.params.width * 0.75) * 0.006
                uncertainty = self.params.heterogeneity * 2.6 + min(0.8, len(self.state.pokes) * 0.13)
                score = best["error"] + angle_penalty + length_penalty + uncertainty
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        0.94 - self.params.heterogeneity * 0.42 - best["error"] / max(30.0, target) - len(self.state.pokes) * 0.018,
                    ),
                )
                candidates.append(
                    Candidate(
                        angle=angle,
                        offset=best["offset"],
                        target_side=side,  # type: ignore[arg-type]
                        portion_a=best["mass"],
                        portion_b=total - best["mass"],
                        error=best["error"],
                        path_length=length,
                        score=score,
                        confidence=confidence,
                        points=points,
                    )
                )

        candidates.sort(key=lambda candidate: candidate.score)
        diverse: list[Candidate] = []
        for candidate in candidates:
            if all(abs(d.angle - candidate.angle) > 0.05235987755982988 or d.target_side != candidate.target_side for d in diverse):
                diverse.append(candidate)
            if len(diverse) >= 3:
                break
        return diverse or candidates[:3]

    def _move_surface(self, surface: SurfaceData) -> SurfaceData:
        return SurfaceData(
            rest=surface.rest.to(self.device),
            current=surface.current.to(self.device),
            density=surface.density.to(self.device),
            viscosity=surface.viscosity.to(self.device),
            stiffness=surface.stiffness.to(self.device),
        )

    def _move_lattice(self, lattice: LatticeData) -> LatticeData:
        return LatticeData(
            rest=lattice.rest.to(self.device),
            current=lattice.current.to(self.device),
            density=lattice.density.to(self.device),
            viscosity=lattice.viscosity.to(self.device),
            stiffness=lattice.stiffness.to(self.device),
            mass_raw=lattice.mass_raw.to(self.device),
            mass=lattice.mass.to(self.device),
        )

    def _move_poke(self, poke: Poke) -> Poke:
        return Poke(
            center=poke.center.to(self.device),
            radius=poke.radius,
            depth=poke.depth,
            push=poke.push.to(self.device),
            amplitude=poke.amplitude,
            held=poke.held,
        )

    def randomize(self, seed: int | None = None) -> None:
        rng = random.Random(self.state.seed if seed is None else seed)
        self.state.seed = rng.randint(1, 9999)
        p = self.params
        p.weight = rng.uniform(190.0, 375.0)
        p.length = rng.uniform(148.0, 211.0)
        p.width = rng.uniform(74.0, 116.0)
        p.thickness = rng.uniform(22.0, 42.0)
        p.taper = rng.uniform(0.38, 0.78)
        p.asymmetry = rng.uniform(-0.12, 0.13)
        p.curvature = rng.uniform(-0.08, 0.08)
        p.irregularity = rng.uniform(0.018, 0.075)
        p.density = rng.uniform(1032.0, 1078.0)
        p.heterogeneity = rng.uniform(0.09, 0.29)
        p.viscosity = rng.uniform(0.34, 0.79)
        p.stiffness = rng.uniform(19.0, 52.0)
        p.recovery = rng.uniform(0.24, 0.72)
        p.friction = rng.uniform(0.30, 0.70)
        p.target_weight = min(p.weight - 45.0, p.weight * rng.uniform(0.40, 0.62))
        p.cut_angle = rng.uniform(-14.0, 14.0)
        p.knife_speed = rng.uniform(30.0, 78.0)

    def build_summary(self, view_mode: ViewMode) -> dict[str, object]:
        candidate = self.state.candidates[0] if self.state.candidates else None
        lattice_nodes = 0 if self.state.lattice is None else int(self.state.lattice.current.shape[0])
        surface_points = 0 if self.state.surface is None else int(self.state.surface.current.shape[0])
        summary: dict[str, object] = {
            "device": str(self.device),
            "view_mode": view_mode,
            "seed": self.state.seed,
            "surface_points": surface_points,
            "lattice_nodes": lattice_nodes,
            "candidate_count": len(self.state.candidates),
            "approved": self.state.approved,
            "cut_done": self.state.cut_done,
            "weight_g": round(self.params.weight, 1),
            "target_weight_g": round(self.params.target_weight, 1),
            "tolerance_g": round(self.params.tolerance, 1),
        }
        if candidate is not None:
            summary["best_candidate"] = {
                "angle_deg": round(float(torch.rad2deg(torch.tensor(candidate.angle))), 2),
                "offset_mm": round(candidate.offset, 2),
                "portion_a_g": round(candidate.portion_a, 2),
                "portion_b_g": round(candidate.portion_b, 2),
                "error_g": round(candidate.error, 2),
                "path_length_mm": round(candidate.path_length, 2),
                "confidence": round(candidate.confidence, 3),
            }
        return summary

    def _blend_hex(self, start: str, end: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        start_rgb = tuple(int(start[i : i + 2], 16) for i in (1, 3, 5))
        end_rgb = tuple(int(end[i : i + 2], 16) for i in (1, 3, 5))
        rgb = tuple(int(round(a + (b - a) * t)) for a, b in zip(start_rgb, end_rgb))
        return "#%02x%02x%02x" % rgb

    def _field_color(self, value: float, min_value: float, max_value: float) -> str:
        if max_value <= min_value:
            return "#e9ad9b"
        t = (value - min_value) / (max_value - min_value)
        stops = ["#2d7fff", "#53d6d0", "#b9eb70", "#ffd65a", "#ff674d"]
        if t < 0.25:
            return self._blend_hex(stops[0], stops[1], t / 0.25)
        if t < 0.5:
            return self._blend_hex(stops[1], stops[2], (t - 0.25) / 0.25)
        if t < 0.75:
            return self._blend_hex(stops[2], stops[3], (t - 0.5) / 0.25)
        return self._blend_hex(stops[3], stops[4], (t - 0.75) / 0.25)

    def _points_to_list(self, points: torch.Tensor) -> list[list[float]]:
        return [[float(row[0]), float(row[1]), float(row[2])] for row in points.detach().cpu()]

    def _candidate_to_dict(self, candidate: Candidate) -> dict[str, object]:
        return {
            "angle": float(candidate.angle),
            "offset": float(candidate.offset),
            "targetSide": candidate.target_side,
            "portionA": float(candidate.portion_a),
            "portionB": float(candidate.portion_b),
            "error": float(candidate.error),
            "pathLength": float(candidate.path_length),
            "score": float(candidate.score),
            "confidence": float(candidate.confidence),
            "points": self._points_to_list(candidate.points),
        }

    def snapshot(self, view_mode: ViewMode = "surface") -> dict[str, object]:
        if self.state.surface is None or self.state.lattice is None:
            self.rebuild()

        assert self.state.surface is not None
        assert self.state.lattice is not None

        surface = self.state.surface
        lattice = self.state.lattice
        return {
            "params": {
                "weight": self.params.weight,
                "length": self.params.length,
                "width": self.params.width,
                "thickness": self.params.thickness,
                "taper": self.params.taper,
                "asymmetry": self.params.asymmetry,
                "curvature": self.params.curvature,
                "irregularity": self.params.irregularity,
                "density": self.params.density,
                "heterogeneity": self.params.heterogeneity,
                "viscosity": self.params.viscosity,
                "stiffness": self.params.stiffness,
                "recovery": self.params.recovery,
                "friction": self.params.friction,
                "targetWeight": self.params.target_weight,
                "tolerance": self.params.tolerance,
                "cutAngle": self.params.cut_angle,
                "knifeSpeed": self.params.knife_speed,
                "bladeDepth": self.params.blade_depth,
                "fingerRadius": self.params.finger_radius,
                "fingerForce": self.params.finger_force,
            },
            "viewMode": view_mode,
            "seed": self.state.seed,
            "approved": self.state.approved,
            "cutDone": self.state.cut_done,
            "selectedCandidate": self.state.selected_candidate,
            "surface": {
                "rest": self._points_to_list(surface.rest),
                "current": self._points_to_list(surface.current),
                "density": [float(v) for v in surface.density.detach().cpu().tolist()],
                "viscosity": [float(v) for v in surface.viscosity.detach().cpu().tolist()],
                "stiffness": [float(v) for v in surface.stiffness.detach().cpu().tolist()],
                "count": int(surface.current.shape[0]),
            },
            "lattice": {
                "rest": self._points_to_list(lattice.rest),
                "current": self._points_to_list(lattice.current),
                "density": [float(v) for v in lattice.density.detach().cpu().tolist()],
                "viscosity": [float(v) for v in lattice.viscosity.detach().cpu().tolist()],
                "stiffness": [float(v) for v in lattice.stiffness.detach().cpu().tolist()],
                "massRaw": [float(v) for v in lattice.mass_raw.detach().cpu().tolist()],
                "mass": [float(v) for v in lattice.mass.detach().cpu().tolist()],
                "count": int(lattice.current.shape[0]),
            },
            "candidates": [self._candidate_to_dict(candidate) for candidate in self.state.candidates],
            "summary": self.build_summary(view_mode),
        }

    def render_svg(self, view_mode: ViewMode) -> str:
        if self.state.surface is None or self.state.lattice is None:
            self.rebuild()

        assert self.state.surface is not None
        assert self.state.lattice is not None

        points = self.state.surface.current if view_mode != "lattice" else self.state.lattice.current
        if points.numel() == 0:
            return "<div>No simulation data.</div>"

        xs = points[:, 0]
        ys = points[:, 1]
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        x_pad = max(8.0, (x_max - x_min) * 0.06)
        y_pad = max(8.0, (y_max - y_min) * 0.08)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        width = 1120
        height = 700
        pad = 48
        plot_w = width - pad * 2
        plot_h = height - pad * 2

        def project(x: float, y: float) -> tuple[float, float]:
            px = pad + (x - x_min) / max(1e-6, x_max - x_min) * plot_w
            py = height - pad - (y - y_min) / max(1e-6, y_max - y_min) * plot_h
            return px, py

        field_name = {
            "surface": None,
            "density": "density",
            "viscosity": "viscosity",
            "stiffness": "stiffness",
            "lattice": "density",
        }[view_mode]
        field_values = None
        if field_name == "density":
            field_values = self.state.surface.density if view_mode != "lattice" else self.state.lattice.density
        elif field_name == "viscosity":
            field_values = self.state.surface.viscosity if view_mode != "lattice" else self.state.lattice.viscosity
        elif field_name == "stiffness":
            field_values = self.state.surface.stiffness if view_mode != "lattice" else self.state.lattice.stiffness

        field_min = float(field_values.min()) if field_values is not None and field_values.numel() else 0.0
        field_max = float(field_values.max()) if field_values is not None and field_values.numel() else 1.0

        draw_points = points
        if draw_points.shape[0] > 2200:
            step = max(1, draw_points.shape[0] // 2200)
            draw_points = draw_points[::step]

        circles: list[str] = []
        for i, point in enumerate(draw_points):
            px, py = project(float(point[0]), float(point[1]))
            if field_values is not None and field_values.numel():
                field_index = min(i * max(1, points.shape[0] // max(1, draw_points.shape[0])), field_values.shape[0] - 1)
                fill = self._field_color(float(field_values[field_index]), field_min, field_max)
                opacity = 0.78 if view_mode != "surface" else 0.88
            else:
                fill = "#e9ad9b"
                opacity = 0.94
            radius = 2.2 if view_mode == "lattice" else 2.0
            circles.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{fill}" fill-opacity="{opacity:.3f}" />')

        path_layers: list[str] = []
        for index, candidate in enumerate(self.state.candidates):
            coords = [project(float(point[0]), float(point[1])) for point in candidate.points]
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
            stroke = "#bdf47c" if index == self.state.selected_candidate else "#ff664f"
            stroke_width = 5 if index == self.state.selected_candidate else 2.5
            path_layers.append(
                f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" stroke-opacity="0.96" />'
            )

        candidate = self.state.candidates[0] if self.state.candidates else None
        best_line = "No candidate yet"
        if candidate is not None:
            best_line = (
                f"Best path: {candidate.confidence * 100:.0f}% confidence, "
                f"{candidate.path_length:.0f} mm, "
                f"{candidate.portion_a:.1f} g / {candidate.portion_b:.1f} g"
            )

        legend = {
            "surface": "surface appearance",
            "density": "density field",
            "viscosity": "viscoelastic damping",
            "stiffness": "relative stiffness",
            "lattice": "particle lattice",
        }[view_mode]

        title = f"FilletTwin Python UI · {html.escape(view_mode.title())} view"
        subtitle = html.escape(best_line)
        stats = html.escape(
            f"seed {self.state.seed} · {legend} · {points.shape[0]} nodes · {len(self.state.candidates)} candidate(s)"
        )

        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">
          <defs>
            <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#111018"/>
              <stop offset="55%" stop-color="#1b1f2a"/>
              <stop offset="100%" stop-color="#120f0f"/>
            </linearGradient>
            <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <rect width="100%" height="100%" fill="url(#bg)" rx="28" />
          <text x="44" y="58" fill="#f6efe9" font-family="Inter, ui-sans-serif, system-ui" font-size="30" font-weight="700">{title}</text>
          <text x="44" y="88" fill="#cdbfb7" font-family="Inter, ui-sans-serif, system-ui" font-size="15">{subtitle}</text>
          <text x="44" y="112" fill="#8f98a6" font-family="Inter, ui-sans-serif, system-ui" font-size="13">{stats}</text>
          <g opacity="0.78">
            <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" rx="22" fill="none" stroke="#3a3f4f" stroke-width="1.2"/>
            <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#465063" stroke-width="1"/>
            <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#465063" stroke-width="1"/>
          </g>
          {''.join(circles)}
          {''.join(path_layers)}
          <g filter="url(#softGlow)">
            <rect x="{width - 306}" y="34" width="260" height="106" rx="18" fill="#171924" fill-opacity="0.88" stroke="#343a49" stroke-width="1"/>
            <text x="{width - 286}" y="66" fill="#e8e3dd" font-family="Inter, ui-sans-serif, system-ui" font-size="18" font-weight="700">Controls active</text>
            <text x="{width - 286}" y="92" fill="#c8d0dd" font-family="Inter, ui-sans-serif, system-ui" font-size="14">CPU now, GPU later</text>
            <text x="{width - 286}" y="116" fill="#9eb2c7" font-family="Inter, ui-sans-serif, system-ui" font-size="13">simulation math only</text>
          </g>
        </svg>
        """.strip()


def _build_app() -> None:
    import gradio as gr

    def render(
        view_mode: ViewMode,
        device: str,
        seed: int,
        weight: float,
        length: float,
        width: float,
        thickness: float,
        taper: float,
        asymmetry: float,
        curvature: float,
        irregularity: float,
        density: float,
        heterogeneity: float,
        viscosity: float,
        stiffness: float,
        recovery: float,
        friction: float,
        target_weight: float,
        tolerance: float,
        cut_angle: float,
        knife_speed: float,
        blade_depth: float,
        finger_radius: float,
        finger_force: float,
    ) -> tuple[str, dict[str, object]]:
        params = Params(
            weight=weight,
            length=length,
            width=width,
            thickness=thickness,
            taper=taper,
            asymmetry=asymmetry,
            curvature=curvature,
            irregularity=irregularity,
            density=density,
            heterogeneity=heterogeneity,
            viscosity=viscosity,
            stiffness=stiffness,
            recovery=recovery,
            friction=friction,
            target_weight=target_weight,
            tolerance=tolerance,
            cut_angle=cut_angle,
            knife_speed=knife_speed,
            blade_depth=blade_depth,
            finger_radius=finger_radius,
            finger_force=finger_force,
        )
        sim = FilletSimulation(params=params, device=device)
        sim.state.seed = seed
        sim.rebuild()
        return sim.render_svg(view_mode), sim.build_summary(view_mode)

    with gr.Blocks(title="FilletTwin Python UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FilletTwin Python UI\nSimulation math in PyTorch, rendered from `python/fillet_sim.py`.")
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                view_mode = gr.Radio(["surface", "density", "viscosity", "stiffness", "lattice"], value="surface", label="View mode")
                device = gr.Dropdown(["cpu"] + (["cuda"] if torch.cuda.is_available() else []), value="cpu", label="Device")
                seed = gr.Slider(1, 9999, value=1, step=1, label="Seed")
                weight = gr.Slider(180, 400, value=285, step=1, label="Weight (g)")
                length = gr.Slider(140, 220, value=182, step=1, label="Length (mm)")
                width = gr.Slider(60, 130, value=96, step=1, label="Width (mm)")
                thickness = gr.Slider(18, 50, value=31, step=1, label="Thickness (mm)")
                taper = gr.Slider(0.2, 1.0, value=0.58, step=0.01, label="Taper")
                asymmetry = gr.Slider(-0.2, 0.2, value=0.05, step=0.01, label="Asymmetry")
                curvature = gr.Slider(-0.1, 0.1, value=0.035, step=0.001, label="Curvature")
                irregularity = gr.Slider(0.0, 0.12, value=0.035, step=0.001, label="Irregularity")
                density = gr.Slider(1000, 1100, value=1055, step=1, label="Density")
                heterogeneity = gr.Slider(0.0, 0.4, value=0.18, step=0.01, label="Heterogeneity")
                viscosity = gr.Slider(0.0, 1.0, value=0.58, step=0.01, label="Viscosity")
                stiffness = gr.Slider(5, 70, value=31, step=1, label="Stiffness")
                recovery = gr.Slider(0.0, 1.0, value=0.42, step=0.01, label="Recovery")
                friction = gr.Slider(0.0, 1.0, value=0.48, step=0.01, label="Friction")
                target_weight = gr.Slider(50, 260, value=160, step=1, label="Target weight (g)")
                tolerance = gr.Slider(0, 20, value=5, step=1, label="Tolerance (g)")
                cut_angle = gr.Slider(-20, 20, value=0, step=1, label="Cut angle (deg)")
                knife_speed = gr.Slider(10, 100, value=48, step=1, label="Knife speed")
                blade_depth = gr.Slider(5, 60, value=36, step=1, label="Blade depth")
                finger_radius = gr.Slider(8, 40, value=24, step=1, label="Finger radius")
                finger_force = gr.Slider(0.1, 1.2, value=0.62, step=0.01, label="Finger force")
                render_button = gr.Button("Render simulation", variant="primary")
            with gr.Column(scale=2, min_width=640):
                svg = gr.HTML(label="Simulation view")
                summary = gr.JSON(label="Simulation summary")

        inputs = [
            view_mode,
            device,
            seed,
            weight,
            length,
            width,
            thickness,
            taper,
            asymmetry,
            curvature,
            irregularity,
            density,
            heterogeneity,
            viscosity,
            stiffness,
            recovery,
            friction,
            target_weight,
            tolerance,
            cut_angle,
            knife_speed,
            blade_depth,
            finger_radius,
            finger_force,
        ]

        render_button.click(render, inputs=inputs, outputs=[svg, summary])
        demo.load(render, inputs=inputs, outputs=[svg, summary])

    demo.launch()


def _coerce_params(payload: dict[str, object]) -> Params:
    return Params(
        weight=float(payload.get("weight", 285.0)),
        length=float(payload.get("length", 182.0)),
        width=float(payload.get("width", 96.0)),
        thickness=float(payload.get("thickness", 31.0)),
        taper=float(payload.get("taper", 0.58)),
        asymmetry=float(payload.get("asymmetry", 0.05)),
        curvature=float(payload.get("curvature", 0.035)),
        irregularity=float(payload.get("irregularity", 0.035)),
        density=float(payload.get("density", 1055.0)),
        heterogeneity=float(payload.get("heterogeneity", 0.18)),
        viscosity=float(payload.get("viscosity", 0.58)),
        stiffness=float(payload.get("stiffness", 31.0)),
        recovery=float(payload.get("recovery", 0.42)),
        friction=float(payload.get("friction", 0.48)),
        target_weight=float(payload.get("targetWeight", payload.get("target_weight", 160.0))),
        tolerance=float(payload.get("tolerance", 5.0)),
        cut_angle=float(payload.get("cutAngle", payload.get("cut_angle", 0.0))),
        knife_speed=float(payload.get("knifeSpeed", payload.get("knife_speed", 48.0))),
        blade_depth=float(payload.get("bladeDepth", payload.get("blade_depth", 36.0))),
        finger_radius=float(payload.get("fingerRadius", payload.get("finger_radius", 24.0))),
        finger_force=float(payload.get("fingerForce", payload.get("finger_force", 0.62))),
    )


def _coerce_pokes(payload: list[dict[str, object]], device: torch.device) -> list[Poke]:
    pokes: list[Poke] = []
    for poke in payload:
        center = poke.get("center", [0.0, 0.0])
        push = poke.get("push", [0.0, 0.0])
        pokes.append(
            Poke(
                center=torch.tensor([float(center[0]), float(center[1])], device=device),
                radius=float(poke.get("radius", 24.0)),
                depth=float(poke.get("depth", 8.0)),
                push=torch.tensor([float(push[0]), float(push[1])], device=device),
                amplitude=float(poke.get("amplitude", 1.0)),
                held=bool(poke.get("held", False)),
            )
        )
    return pokes


def create_api_app() -> FastAPI:
    app = FastAPI(title="FilletTwin Python API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/snapshot")
    def snapshot(payload: dict[str, object]) -> dict[str, object]:
        params = _coerce_params(dict(payload.get("params", {})))
        view_mode = payload.get("viewMode", "surface")
        seed = int(payload.get("seed", 1))
        approved = bool(payload.get("approved", False))
        cut_done = bool(payload.get("cutDone", False))
        selected_candidate = int(payload.get("selectedCandidate", 0))
        device_name = str(payload.get("device", "cpu"))
        device = torch.device(device_name)

        sim = FilletSimulation(params=params, device=device)
        sim.state.seed = seed
        sim.rebuild()
        sim.state.approved = approved
        sim.state.cut_done = cut_done
        poke_payload = payload.get("pokes", [])
        if isinstance(poke_payload, list):
            sim.state.pokes = _coerce_pokes(poke_payload, device)
            sim.update_geometry_positions()
            sim.state.candidates = sim.optimize_paths()
        sim.state.selected_candidate = min(selected_candidate, max(0, len(sim.state.candidates) - 1))
        return sim.snapshot(view_mode=str(view_mode))

    # Serve the exact Three.js application built from src/main.ts.  This keeps
    # the visual UI in the browser while making this Python process the sole
    # simulation/API backbone and the single application entry point.
    web_root = Path(__file__).resolve().parents[1] / "dist"
    if (web_root / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_root, html=True), name="web")

    return app


def _serve_api() -> None:
    api_app = create_api_app()
    uvicorn.run(api_app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FilletTwin Python simulation backend")
    parser.add_argument("--gradio", action="store_true", help="Launch the standalone Gradio demo instead of the API server")
    args = parser.parse_args()
    if args.gradio:
        _build_app()
    else:
        _serve_api()
