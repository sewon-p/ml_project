"""Sensor fusion: GPS lat/lon → local XY, accelerometer → brake detection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Earth radius in meters (WGS-84 mean)
_EARTH_RADIUS = 6_371_000.0

# Brake detection threshold: longitudinal deceleration below this → brake=1
_BRAKE_DECEL_THRESHOLD = -1.5  # m/s²


@dataclass
class ReferencePoint:
    """Origin for equirectangular projection (first GPS fix)."""

    lat: float  # degrees
    lon: float  # degrees
    cos_lat: float = field(init=False)

    def __post_init__(self) -> None:
        self.cos_lat = math.cos(math.radians(self.lat))


def gps_to_xy(
    lat: float,
    lon: float,
    ref: ReferencePoint,
) -> tuple[float, float]:
    """Convert GPS (lat, lon) to local (x, y) meters using equirectangular projection.

    Valid within ~10 km of the reference point.
    Returns (x_east, y_north) in meters.
    """
    dlat = math.radians(lat - ref.lat)
    dlon = math.radians(lon - ref.lon)
    x_east = dlon * _EARTH_RADIUS * ref.cos_lat
    y_north = dlat * _EARTH_RADIUS
    return x_east, y_north


def rotate_to_heading(
    x_east: float,
    y_north: float,
    heading_deg: float,
) -> tuple[float, float]:
    """Rotate (east, north) so that X=forward (heading direction), Y=lateral.

    heading_deg: clockwise from north (GPS convention).
    """
    theta = math.radians(heading_deg)
    cos_h = math.cos(theta)
    sin_h = math.sin(theta)
    # Forward = along heading, lateral = perpendicular right
    x_fwd = x_east * sin_h + y_north * cos_h
    y_lat = x_east * cos_h - y_north * sin_h
    return x_fwd, y_lat


def detect_brake(accel_x: float) -> float:
    """Detect braking from longitudinal acceleration.

    Args:
        accel_x: Longitudinal acceleration (m/s²), negative = deceleration.

    Returns:
        1.0 if braking, 0.0 otherwise.
    """
    return 1.0 if accel_x < _BRAKE_DECEL_THRESHOLD else 0.0


@dataclass
class SensorFusion:
    """Stateful sensor fusion for a single session.

    Converts raw GPS + accelerometer readings into the 6-channel FCD format
    expected by the ML pipeline: (time, x, y, speed, brake) where x/y are
    forward/lateral in meters relative to the first GPS fix.
    """

    ref: ReferencePoint | None = None
    _prev_x: float = 0.0
    _prev_y: float = 0.0

    def process(
        self,
        *,
        lat: float,
        lon: float,
        speed: float,
        heading: float = 0.0,
        accel_x: float = 0.0,
        accel_y: float = 0.0,
        accel_z: float = 0.0,
        timestamp: float = 0.0,
    ) -> dict[str, float]:
        """Process one sensor reading and return an FCD record dict.

        Args:
            lat, lon: GPS coordinates (degrees).
            speed: GPS speed (m/s).
            heading: GPS heading (degrees clockwise from north).
            accel_x: Longitudinal acceleration (m/s²).
            accel_y: Lateral acceleration (m/s²).
            accel_z: Vertical acceleration (m/s²).
            timestamp: Unix timestamp or elapsed seconds.

        Returns:
            Dict with keys: time, x, y, speed, brake.
        """
        # Initialize reference point on first GPS fix
        if self.ref is None:
            self.ref = ReferencePoint(lat=lat, lon=lon)

        # GPS → local XY
        x_east, y_north = gps_to_xy(lat, lon, self.ref)

        # Rotate so X=forward, Y=lateral
        x_fwd, y_lat = rotate_to_heading(x_east, y_north, heading)

        # Brake detection from longitudinal deceleration
        brake = detect_brake(accel_x)

        self._prev_x = x_fwd
        self._prev_y = y_lat

        return {
            "time": timestamp,
            "x": x_fwd,
            "y": y_lat,
            "speed": speed,
            "brake": brake,
        }


def compute_velocity_components(
    speed: float,
    heading_deg: float,
) -> tuple[float, float]:
    """Compute VX (forward) and VY (lateral) from speed and heading.

    For the 6-channel model: VX ≈ speed, VY ≈ 0 when traveling straight.
    In practice, VX and VY are derived from consecutive position differences,
    but this provides initial estimates from GPS data.
    """
    # In the rotated frame, VX = speed (forward), VY ≈ 0
    # GPS heading jitter can cause small VY components
    return speed, 0.0


def batch_fuse(
    records: list[dict],
) -> list[dict[str, float]]:
    """Fuse a batch of raw sensor records into FCD format.

    Each record should have: lat, lon, speed, heading, accel_x, accel_y,
    accel_z, timestamp.

    Returns list of dicts with: time, x, y, speed, brake.
    """
    fusion = SensorFusion()
    return [
        fusion.process(
            lat=r.get("lat", 0.0),
            lon=r.get("lon", 0.0),
            speed=r.get("speed", 0.0),
            heading=r.get("heading", 0.0),
            accel_x=r.get("accel_x", 0.0),
            accel_y=r.get("accel_y", 0.0),
            accel_z=r.get("accel_z", 0.0),
            timestamp=r.get("timestamp", float(i)),
        )
        for i, r in enumerate(records)
    ]
