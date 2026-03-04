"""Sensor fusion: GPS lat/lon → local XY, accelerometer → brake, Kalman Filter smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Earth radius in meters (WGS-84 mean)
_EARTH_RADIUS = 6_371_000.0

# Brake detection threshold: longitudinal deceleration below this → brake=1
_BRAKE_DECEL_THRESHOLD = -1.5  # m/s²

# Kalman Filter default parameters
_GPS_POSITION_NOISE = 5.0  # GPS position noise σ (meters)
_GPS_SPEED_NOISE = 1.0  # GPS speed noise σ (m/s)
_ACCEL_PROCESS_NOISE = 2.0  # accelerometer process noise σ (m/s²)
_DT = 1.0  # time step (seconds) — 1 Hz sampling


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


# ---------------------------------------------------------------------------
# Kalman Filter for GPS + Accelerometer fusion
# ---------------------------------------------------------------------------


class KalmanFilter2D:
    """2D Kalman Filter for fusing GPS position/speed with accelerometer.

    State vector: [x, vx, y, vy] (position and velocity in east/north frame)

    Prediction step uses accelerometer as control input.
    Update step uses GPS position (x, y) and GPS speed (decomposed into vx, vy).
    """

    def __init__(
        self,
        gps_pos_noise: float = _GPS_POSITION_NOISE,
        gps_speed_noise: float = _GPS_SPEED_NOISE,
        accel_noise: float = _ACCEL_PROCESS_NOISE,
        dt: float = _DT,
    ) -> None:
        self.dt = dt

        # State: [x, vx, y, vy]
        self.x = np.zeros(4)

        # State covariance — start with high uncertainty
        self.P = np.eye(4) * 100.0

        # State transition matrix (constant velocity model)
        self.F = np.array(
            [
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1],
            ]
        )

        # Control input matrix: accelerometer [ax, ay] → state change
        self.B = np.array(
            [
                [0.5 * dt**2, 0],
                [dt, 0],
                [0, 0.5 * dt**2],
                [0, dt],
            ]
        )

        # Process noise (accelerometer uncertainty drives state uncertainty)
        q = accel_noise**2
        self.Q = np.array(
            [
                [dt**4 / 4, dt**3 / 2, 0, 0],
                [dt**3 / 2, dt**2, 0, 0],
                [0, 0, dt**4 / 4, dt**3 / 2],
                [0, 0, dt**3 / 2, dt**2],
            ]
        ) * q

        # Measurement matrix: observe [x, vx, y, vy] from GPS
        self.H = np.eye(4)

        # Measurement noise: GPS position + GPS speed
        self.R = np.diag(
            [gps_pos_noise**2, gps_speed_noise**2, gps_pos_noise**2, gps_speed_noise**2]
        )

        self._initialized = False

    def initialize(self, x: float, vx: float, y: float, vy: float) -> None:
        """Set initial state from first GPS observation."""
        self.x = np.array([x, vx, y, vy])
        self._initialized = True

    def predict(self, accel_east: float = 0.0, accel_north: float = 0.0) -> None:
        """Prediction step using accelerometer as control input."""
        u = np.array([accel_east, accel_north])
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, x_obs: float, vx_obs: float, y_obs: float, vy_obs: float) -> None:
        """Update step using GPS observation."""
        z = np.array([x_obs, vx_obs, y_obs, vy_obs])
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        eye = np.eye(4)
        self.P = (eye - K @ self.H) @ self.P

    @property
    def state(self) -> tuple[float, float, float, float]:
        """Return filtered (x, vx, y, vy)."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# SensorFusion (stateful, per-session)
# ---------------------------------------------------------------------------


@dataclass
class SensorFusion:
    """Stateful sensor fusion for a single session.

    Converts raw GPS + accelerometer readings into the 6-channel FCD format
    expected by the ML pipeline: (time, x, y, speed, brake) where x/y are
    forward/lateral in meters relative to the first GPS fix.

    Uses a Kalman Filter to smooth GPS noise and fuse accelerometer data.
    Set use_kalman=False to disable Kalman filtering (raw GPS passthrough).
    """

    ref: ReferencePoint | None = None
    use_kalman: bool = True
    _kf: KalmanFilter2D | None = field(default=None, init=False, repr=False)
    _prev_heading: float = field(default=0.0, init=False, repr=False)

    def _ensure_kf(self) -> KalmanFilter2D:
        if self._kf is None:
            self._kf = KalmanFilter2D()
        return self._kf

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

        # GPS → local XY (east/north frame)
        x_east, y_north = gps_to_xy(lat, lon, self.ref)

        # Decompose GPS speed into east/north components using heading
        heading_rad = math.radians(heading)
        vx_east = speed * math.sin(heading_rad)
        vy_north = speed * math.cos(heading_rad)

        # Decompose accelerometer into east/north frame
        # accel_x = longitudinal (along heading), accel_y = lateral (perpendicular)
        accel_east = accel_x * math.sin(heading_rad) + accel_y * math.cos(heading_rad)
        accel_north = accel_x * math.cos(heading_rad) - accel_y * math.sin(heading_rad)

        if self.use_kalman:
            kf = self._ensure_kf()
            if not kf.is_initialized:
                kf.initialize(x_east, vx_east, y_north, vy_north)
            else:
                # Predict with accelerometer, then update with GPS
                kf.predict(accel_east, accel_north)
                kf.update(x_east, vx_east, y_north, vy_north)

            filt_x, filt_vx, filt_y, filt_vy = kf.state
            filt_speed = math.sqrt(filt_vx**2 + filt_vy**2)
        else:
            filt_x, filt_y = x_east, y_north
            filt_speed = speed

        # Rotate filtered position to heading frame (X=forward, Y=lateral)
        x_fwd, y_lat = rotate_to_heading(filt_x, filt_y, heading)

        # Brake detection from longitudinal deceleration
        brake = detect_brake(accel_x)

        self._prev_heading = heading

        return {
            "time": timestamp,
            "x": x_fwd,
            "y": y_lat,
            "speed": filt_speed,
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
    use_kalman: bool = True,
) -> list[dict[str, float]]:
    """Fuse a batch of raw sensor records into FCD format.

    Each record should have: lat, lon, speed, heading, accel_x, accel_y,
    accel_z, timestamp.

    Args:
        records: List of raw sensor readings.
        use_kalman: Whether to apply Kalman Filter smoothing.

    Returns list of dicts with: time, x, y, speed, brake.
    """
    fusion = SensorFusion(use_kalman=use_kalman)
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
