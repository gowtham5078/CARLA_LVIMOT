import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanTrack:
    """
    Constant-velocity Kalman track.

    State:
        [x, y, z, vx, vy, vz]
    """

    def __init__(self, track_id, observation, timestamp):
        self.track_id = track_id
        self.class_name = observation["type"]

        position = np.asarray(
            observation["location"],
            dtype=np.float64
        )

        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = position

        # Initial velocity is unknown.
        self.state[3:] = 0.0

        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 1.0
        self.P[3:, 3:] *= 10.0

        self.last_timestamp = float(timestamp)

        self.age = 1
        self.hits = 1
        self.missed = 0

        self.history = [
            {
                "timestamp": float(timestamp),
                "position": position.copy(),
                "velocity": self.state[3:].copy(),
            }
        ]

    def predict(self, dt):
        """
        Constant velocity prediction.
        """

        dt = max(float(dt), 1e-6)

        F = np.eye(6, dtype=np.float64)

        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Process noise.
        q = 1.0

        Q = np.zeros((6, 6), dtype=np.float64)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        Q[:3, :3] = np.eye(3) * (dt4 / 4.0) * q
        Q[:3, 3:] = np.eye(3) * (dt3 / 2.0) * q
        Q[3:, :3] = np.eye(3) * (dt3 / 2.0) * q
        Q[3:, 3:] = np.eye(3) * dt2 * q

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        return self.state.copy()

    def update(self, observation, timestamp):
        """
        Position measurement update.
        """

        z = np.asarray(
            observation["location"],
            dtype=np.float64
        )

        H = np.zeros((3, 6), dtype=np.float64)
        H[:, :3] = np.eye(3)

        R = np.eye(3, dtype=np.float64) * 0.25

        innovation = z - H @ self.state

        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation

        I = np.eye(6)

        self.P = (I - K @ H) @ self.P

        self.last_timestamp = float(timestamp)

        self.age += 1
        self.hits += 1
        self.missed = 0

        self.history.append(
            {
                "timestamp": float(timestamp),
                "position": self.state[:3].copy(),
                "velocity": self.state[3:].copy(),
            }
        )

    def mark_missed(self):
        self.missed += 1
        self.age += 1

    @property
    def position(self):
        return self.state[:3]

    @property
    def velocity(self):
        return self.state[3:]


class CarlaMultiObjectTracker:
    """
    Multi-object tracker using:

        Kalman prediction
        +
        Hungarian assignment
        +
        Euclidean distance gating
    """

    def __init__(
        self,
        distance_threshold=5.0,
        max_missed=5,
        min_hits=1,
    ):
        self.distance_threshold = distance_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits

        self.tracks = []
        self.next_track_id = 0

    def _create_track(self, observation, timestamp):

        track = KalmanTrack(
            track_id=self.next_track_id,
            observation=observation,
            timestamp=timestamp,
        )

        self.next_track_id += 1

        self.tracks.append(track)

        return track

    def _build_cost_matrix(self, observations):

        if not self.tracks or not observations:
            return np.empty(
                (len(self.tracks), len(observations))
            )

        cost = np.full(
            (len(self.tracks), len(observations)),
            np.inf,
            dtype=np.float64,
        )

        for i, track in enumerate(self.tracks):

            for j, observation in enumerate(observations):

                # Only compare objects of the same class.
                if track.class_name != observation["type"]:
                    continue

                position = np.asarray(
                    observation["location"],
                    dtype=np.float64,
                )

                distance = np.linalg.norm(
                    track.position - position
                )

                if distance <= self.distance_threshold:
                    cost[i, j] = distance

        return cost

    def update(self, observations, timestamp):

        observations = list(observations)

        timestamp = float(timestamp)

        # --------------------------------------------------
        # 1. Predict existing tracks
        # --------------------------------------------------

        for track in self.tracks:

            dt = timestamp - track.last_timestamp

            track.predict(dt)

        # --------------------------------------------------
        # 2. Data association
        # --------------------------------------------------

        cost_matrix = self._build_cost_matrix(observations)

        matched_tracks = set()
        matched_observations = set()

        if (
            len(self.tracks) > 0
            and len(observations) > 0
        ):

            rows, cols = linear_sum_assignment(
                np.where(
                    np.isfinite(cost_matrix),
                    cost_matrix,
                    1e9,
                )
            )

            for row, col in zip(rows, cols):

                if not np.isfinite(cost_matrix[row, col]):
                    continue

                matched_tracks.add(row)
                matched_observations.add(col)

                self.tracks[row].update(
                    observations[col],
                    timestamp,
                )

        # --------------------------------------------------
        # 3. Mark unmatched tracks
        # --------------------------------------------------

        for i, track in enumerate(self.tracks):

            if i not in matched_tracks:
                track.mark_missed()

        # --------------------------------------------------
        # 4. Create new tracks
        # --------------------------------------------------

        for j, observation in enumerate(observations):

            if j not in matched_observations:

                self._create_track(
                    observation,
                    timestamp,
                )

        # --------------------------------------------------
        # 5. Delete stale tracks
        # --------------------------------------------------

        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.max_missed
        ]

        # --------------------------------------------------
        # 6. Return active tracks
        # --------------------------------------------------

        return self.get_tracks()

    def get_tracks(self):

        results = []

        for track in self.tracks:

            if track.hits < self.min_hits:
                continue

            results.append(
                {
                    "track_id": track.track_id,
                    "class": track.class_name,
                    "position": track.position.tolist(),
                    "velocity": track.velocity.tolist(),
                    "age": track.age,
                    "hits": track.hits,
                    "missed": track.missed,
                }
            )

        return results
