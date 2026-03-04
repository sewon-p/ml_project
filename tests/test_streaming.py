"""Tests for the streaming pipeline: sensor fusion + session window."""

from __future__ import annotations

from src.streaming.fusion import (
    KalmanFilter2D,
    ReferencePoint,
    SensorFusion,
    batch_fuse,
    detect_brake,
    gps_to_xy,
    rotate_to_heading,
)

# ---------------------------------------------------------------------------
# GPS → XY conversion
# ---------------------------------------------------------------------------


class TestGpsToXY:
    """Test equirectangular GPS → local XY conversion."""

    def test_origin_is_zero(self):
        ref = ReferencePoint(lat=37.5665, lon=126.978)
        x, y = gps_to_xy(37.5665, 126.978, ref)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_north_displacement(self):
        """Moving ~111m north (0.001° lat) should give y ≈ 111m."""
        ref = ReferencePoint(lat=37.5665, lon=126.978)
        _, y = gps_to_xy(37.5675, 126.978, ref)
        assert 100 < y < 120  # ~111m per 0.001°

    def test_east_displacement(self):
        """Moving east should give positive x."""
        ref = ReferencePoint(lat=37.5665, lon=126.978)
        x, _ = gps_to_xy(37.5665, 126.979, ref)
        assert x > 50  # should be positive and substantial

    def test_cos_lat_factor(self):
        """East displacement at equator should be larger than at high latitudes."""
        ref_equator = ReferencePoint(lat=0.0, lon=0.0)
        ref_high = ReferencePoint(lat=60.0, lon=0.0)
        x_eq, _ = gps_to_xy(0.0, 0.001, ref_equator)
        x_hi, _ = gps_to_xy(60.0, 0.001, ref_high)
        assert x_eq > x_hi  # cos(0) > cos(60)


# ---------------------------------------------------------------------------
# Heading rotation
# ---------------------------------------------------------------------------


class TestRotateToHeading:
    """Test rotation from (east, north) to (forward, lateral)."""

    def test_north_heading(self):
        """Heading=0 (north): forward=north, lateral=east."""
        x_fwd, y_lat = rotate_to_heading(0.0, 10.0, 0.0)
        assert abs(x_fwd - 10.0) < 1e-6  # north → forward
        assert abs(y_lat) < 1e-6

    def test_east_heading(self):
        """Heading=90 (east): forward=east."""
        x_fwd, y_lat = rotate_to_heading(10.0, 0.0, 90.0)
        assert abs(x_fwd - 10.0) < 1e-6

    def test_south_heading(self):
        """Heading=180 (south): going south means -north is forward."""
        x_fwd, _ = rotate_to_heading(0.0, -10.0, 180.0)
        assert abs(x_fwd - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Brake detection
# ---------------------------------------------------------------------------


class TestBrakeDetection:
    def test_hard_brake(self):
        assert detect_brake(-2.0) == 1.0

    def test_mild_decel(self):
        assert detect_brake(-1.0) == 0.0

    def test_acceleration(self):
        assert detect_brake(2.0) == 0.0

    def test_threshold_exact(self):
        assert detect_brake(-1.5) == 0.0  # not strictly less than
        assert detect_brake(-1.51) == 1.0


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------


class TestKalmanFilter2D:
    def test_initialization(self):
        kf = KalmanFilter2D()
        kf.initialize(x=0.0, vx=10.0, y=0.0, vy=0.0)
        assert kf.is_initialized
        x, vx, y, vy = kf.state
        assert x == 0.0
        assert vx == 10.0

    def test_predict_constant_velocity(self):
        """Without acceleration, position should advance by velocity * dt."""
        kf = KalmanFilter2D(dt=1.0)
        kf.initialize(x=0.0, vx=10.0, y=0.0, vy=5.0)
        kf.predict(accel_east=0.0, accel_north=0.0)
        x, vx, y, vy = kf.state
        assert abs(x - 10.0) < 1e-6
        assert abs(y - 5.0) < 1e-6

    def test_predict_with_acceleration(self):
        """Acceleration should increase velocity and add 0.5*a*t² to position."""
        kf = KalmanFilter2D(dt=1.0)
        kf.initialize(x=0.0, vx=0.0, y=0.0, vy=0.0)
        kf.predict(accel_east=2.0, accel_north=0.0)
        x, vx, y, vy = kf.state
        assert abs(x - 1.0) < 1e-6  # 0.5 * 2.0 * 1.0²
        assert abs(vx - 2.0) < 1e-6  # 0 + 2.0 * 1.0

    def test_update_corrects_prediction(self):
        """GPS observation should pull the state toward the measurement."""
        kf = KalmanFilter2D(dt=1.0)
        kf.initialize(x=0.0, vx=10.0, y=0.0, vy=0.0)
        kf.predict()
        # Predicted x=10, but GPS says x=12
        kf.update(x_obs=12.0, vx_obs=10.0, y_obs=0.0, vy_obs=0.0)
        x, _, _, _ = kf.state
        # Should be between predicted (10) and observed (12)
        assert 10.0 < x < 12.0

    def test_noise_smoothing(self):
        """Kalman filter should produce smoother trajectory than raw GPS."""
        import math

        kf = KalmanFilter2D(gps_pos_noise=5.0, gps_speed_noise=1.0, dt=1.0)
        speed = 10.0  # m/s, heading east

        raw_positions = []
        filtered_positions = []

        for i in range(50):
            true_x = speed * i
            # Add GPS noise
            noise = 3.0 * math.sin(i * 0.7) + 1.5 * math.cos(i * 1.3)
            gps_x = true_x + noise

            if not kf.is_initialized:
                kf.initialize(gps_x, speed, 0.0, 0.0)
            else:
                kf.predict(accel_east=0.0, accel_north=0.0)
                kf.update(gps_x, speed, 0.0, 0.0)

            raw_positions.append(gps_x)
            filt_x, _, _, _ = kf.state
            filtered_positions.append(filt_x)

        # Compute jitter (consecutive differences variance) — filtered should be smoother
        raw_diffs = [raw_positions[i + 1] - raw_positions[i] for i in range(len(raw_positions) - 1)]
        filt_diffs = [
            filtered_positions[i + 1] - filtered_positions[i]
            for i in range(len(filtered_positions) - 1)
        ]
        raw_var = sum((d - speed) ** 2 for d in raw_diffs) / len(raw_diffs)
        filt_var = sum((d - speed) ** 2 for d in filt_diffs) / len(filt_diffs)
        assert filt_var < raw_var  # filtered is smoother


# ---------------------------------------------------------------------------
# SensorFusion stateful processing
# ---------------------------------------------------------------------------


class TestSensorFusion:
    def test_first_point_sets_reference(self):
        fusion = SensorFusion(use_kalman=False)
        assert fusion.ref is None
        result = fusion.process(lat=37.5665, lon=126.978, speed=10.0, timestamp=0.0)
        assert fusion.ref is not None
        assert abs(result["x"]) < 1e-6
        assert abs(result["y"]) < 1e-6

    def test_sequential_points(self):
        fusion = SensorFusion(use_kalman=False)
        r1 = fusion.process(lat=37.5665, lon=126.978, speed=10.0, timestamp=0.0)
        r2 = fusion.process(lat=37.5675, lon=126.978, speed=10.0, heading=0.0, timestamp=1.0)
        # Second point is north → x (forward) should increase when heading=0
        assert r2["x"] > r1["x"]

    def test_speed_passthrough_raw(self):
        fusion = SensorFusion(use_kalman=False)
        result = fusion.process(lat=0.0, lon=0.0, speed=15.5, timestamp=0.0)
        assert result["speed"] == 15.5

    def test_kalman_smooths_speed(self):
        """With Kalman filter, speed should be smoothed (not raw passthrough)."""
        fusion = SensorFusion(use_kalman=True)
        # First point initializes KF
        fusion.process(lat=37.5665, lon=126.978, speed=10.0, heading=0.0, timestamp=0.0)
        # Second point with very different speed → KF should smooth it
        r = fusion.process(lat=37.5666, lon=126.978, speed=20.0, heading=0.0, timestamp=1.0)
        # Kalman should pull speed between 10 and 20
        assert 10.0 < r["speed"] < 20.0

    def test_brake_from_accel(self):
        fusion = SensorFusion(use_kalman=False)
        result = fusion.process(lat=0.0, lon=0.0, speed=10.0, accel_x=-3.0, timestamp=0.0)
        assert result["brake"] == 1.0

    def test_no_brake(self):
        fusion = SensorFusion(use_kalman=False)
        result = fusion.process(lat=0.0, lon=0.0, speed=10.0, accel_x=1.0, timestamp=0.0)
        assert result["brake"] == 0.0

    def test_kalman_enabled_by_default(self):
        fusion = SensorFusion()
        assert fusion.use_kalman is True


# ---------------------------------------------------------------------------
# Batch fusion
# ---------------------------------------------------------------------------


class TestBatchFuse:
    def test_empty(self):
        assert batch_fuse([]) == []

    def test_multiple_records_raw(self):
        records = [
            {
                "lat": 37.5665,
                "lon": 126.978,
                "speed": 10.0,
                "heading": 0.0,
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 0.0,
            },
            {
                "lat": 37.5666,
                "lon": 126.978,
                "speed": 11.0,
                "heading": 0.0,
                "accel_x": -2.0,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 1.0,
            },
            {
                "lat": 37.5667,
                "lon": 126.978,
                "speed": 12.0,
                "heading": 0.0,
                "accel_x": 0.5,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 2.0,
            },
        ]
        results = batch_fuse(records, use_kalman=False)
        assert len(results) == 3
        assert results[0]["time"] == 0.0
        assert results[1]["brake"] == 1.0  # hard decel
        assert results[2]["brake"] == 0.0
        # Positions should increase (moving north)
        assert results[1]["x"] > results[0]["x"]
        assert results[2]["x"] > results[1]["x"]

    def test_multiple_records_with_kalman(self):
        records = [
            {
                "lat": 37.5665,
                "lon": 126.978,
                "speed": 10.0,
                "heading": 0.0,
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 0.0,
            },
            {
                "lat": 37.5666,
                "lon": 126.978,
                "speed": 11.0,
                "heading": 0.0,
                "accel_x": -2.0,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 1.0,
            },
            {
                "lat": 37.5667,
                "lon": 126.978,
                "speed": 12.0,
                "heading": 0.0,
                "accel_x": 0.5,
                "accel_y": 0.0,
                "accel_z": 0.0,
                "timestamp": 2.0,
            },
        ]
        results = batch_fuse(records, use_kalman=True)
        assert len(results) == 3
        assert results[1]["brake"] == 1.0
        assert results[2]["brake"] == 0.0
        # Kalman-filtered positions should still be monotonically increasing
        assert results[1]["x"] > results[0]["x"]

    def test_keys_present(self):
        records = [{"lat": 0, "lon": 0, "speed": 5, "timestamp": 0}]
        result = batch_fuse(records, use_kalman=False)[0]
        assert set(result.keys()) == {"time", "x", "y", "speed", "brake"}


# ---------------------------------------------------------------------------
# SessionState (consumer)
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_buffer_accumulation(self):
        from src.streaming.consumer import SessionState

        session = SessionState(session_id="test-1")
        assert not session.ready
        for i in range(300):
            session.add({"time": float(i), "x": 0, "y": 0, "speed": 10, "brake": 0})
        assert session.ready
        assert len(session.get_window()) == 300

    def test_sliding_window(self):
        from src.streaming.consumer import SessionState

        session = SessionState(session_id="test-2")
        for i in range(350):
            session.add({"time": float(i), "x": 0, "y": 0, "speed": 10, "brake": 0})
        window = session.get_window()
        assert len(window) == 300
        # Should contain the most recent 300 records
        assert window[0]["time"] == 50.0
        assert window[-1]["time"] == 349.0

    def test_not_ready_until_full(self):
        from src.streaming.consumer import SessionState

        session = SessionState(session_id="test-3")
        for i in range(299):
            session.add({"time": float(i), "x": 0, "y": 0, "speed": 10, "brake": 0})
        assert not session.ready
        session.add({"time": 299.0, "x": 0, "y": 0, "speed": 10, "brake": 0})
        assert session.ready
