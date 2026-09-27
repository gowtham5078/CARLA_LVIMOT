import numpy as np


class IMUPreintegrator:

    def __init__(self):
        pass

    def integrate(self, motion_k, motion_k1):

        dt = motion_k["dt"]

        # Acceleration at frame k
        a_k = np.asarray(
            motion_k["acceleration"],
            dtype=np.float64
        )

        # Acceleration at frame k+1
        a_k1 = np.asarray(
            motion_k1["acceleration"],
            dtype=np.float64
        )

        # Angular velocity at frame k
        w_k = np.asarray(
            motion_k["gyro"],
            dtype=np.float64
        )

        # Angular velocity at frame k+1
        w_k1 = np.asarray(
            motion_k1["gyro"],
            dtype=np.float64
        )

        # Average acceleration
        acceleration = 0.5 * (a_k + a_k1)

        # Average angular velocity
        angular_velocity = 0.5 * (w_k + w_k1)

        # Velocity increment
        delta_v = acceleration * dt

        # Position increment
        delta_p = 0.5 * acceleration * (dt ** 2)

        # Small-angle rotation increment
        delta_theta = angular_velocity * dt

        # Convert small-angle rotation vector
        # into an approximate rotation matrix
        wx, wy, wz = delta_theta

        delta_R = np.array([
            [1.0, -wz,  wy],
            [wz,  1.0, -wx],
            [-wy, wx,  1.0]
        ])

        return {
            "delta_p": delta_p,
            "delta_v": delta_v,
            "delta_theta": delta_theta,
            "delta_R": delta_R
        }
