"""Tests for the streaming pipeline: sensor fusion + session window."""

from __future__ import annotations

from src.streaming.fusion import (
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
# SensorFusion stateful processing
# ---------------------------------------------------------------------------


class TestSensorFusion:
    def test_first_point_sets_reference(self):
        fusion = SensorFusion()
        assert fusion.ref is None
        result = fusion.process(lat=37.5665, lon=126.978, speed=10.0, timestamp=0.0)
        assert fusion.ref is not None
        assert abs(result["x"]) < 1e-6
        assert abs(result["y"]) < 1e-6

    def test_sequential_points(self):
        fusion = SensorFusion()
        r1 = fusion.process(lat=37.5665, lon=126.978, speed=10.0, timestamp=0.0)
        r2 = fusion.process(lat=37.5675, lon=126.978, speed=10.0, heading=0.0, timestamp=1.0)
        # Second point is north → x (forward) should increase when heading=0
        assert r2["x"] > r1["x"]

    def test_speed_passthrough(self):
        fusion = SensorFusion()
        result = fusion.process(lat=0.0, lon=0.0, speed=15.5, timestamp=0.0)
        assert result["speed"] == 15.5

    def test_brake_from_accel(self):
        fusion = SensorFusion()
        result = fusion.process(lat=0.0, lon=0.0, speed=10.0, accel_x=-3.0, timestamp=0.0)
        assert result["brake"] == 1.0

    def test_no_brake(self):
        fusion = SensorFusion()
        result = fusion.process(lat=0.0, lon=0.0, speed=10.0, accel_x=1.0, timestamp=0.0)
        assert result["brake"] == 0.0


# ---------------------------------------------------------------------------
# Batch fusion
# ---------------------------------------------------------------------------


class TestBatchFuse:
    def test_empty(self):
        assert batch_fuse([]) == []

    def test_multiple_records(self):
        records = [
            {"lat": 37.5665, "lon": 126.978, "speed": 10.0, "heading": 0.0,
             "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0, "timestamp": 0.0},
            {"lat": 37.5666, "lon": 126.978, "speed": 11.0, "heading": 0.0,
             "accel_x": -2.0, "accel_y": 0.0, "accel_z": 0.0, "timestamp": 1.0},
            {"lat": 37.5667, "lon": 126.978, "speed": 12.0, "heading": 0.0,
             "accel_x": 0.5, "accel_y": 0.0, "accel_z": 0.0, "timestamp": 2.0},
        ]
        results = batch_fuse(records)
        assert len(results) == 3
        assert results[0]["time"] == 0.0
        assert results[1]["brake"] == 1.0  # hard decel
        assert results[2]["brake"] == 0.0
        # Positions should increase (moving north)
        assert results[1]["x"] > results[0]["x"]
        assert results[2]["x"] > results[1]["x"]

    def test_keys_present(self):
        records = [{"lat": 0, "lon": 0, "speed": 5, "timestamp": 0}]
        result = batch_fuse(records)[0]
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
