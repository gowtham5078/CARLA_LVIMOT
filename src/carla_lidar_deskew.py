import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class CarlaLiDARDeskewer:

    def __init__(self, scan_duration=0.1):
        """
        Approximate CARLA LiDAR deskewer.

        CARLA LiDAR is configured at 10 Hz, so one complete
        LiDAR rotation takes approximately 0.1 seconds.

        The saved CARLA .bin files contain only:

            [x, y, z, intensity]

        Therefore individual point timestamps are not available.
        Point acquisition time is currently approximated.
        """

        self.scan_duration = scan_duration

    # --------------------------------------------------------
    # Point timing
    # --------------------------------------------------------

    def create_point_times(self, num_points):
        """
        Approximate acquisition time for each point.

        Current approximation:
            first point -> 0 seconds
            last point  -> scan_duration
        """

        return np.linspace(
            0.0,
            self.scan_duration,
            num_points
        )

    # --------------------------------------------------------
    # Pose interpolation
    # --------------------------------------------------------

    def interpolate_pose(
        self,
        alpha,
        translation_start,
        translation_end,
        rotation_start,
        rotation_end
    ):
        """
        Interpolate vehicle pose.

        alpha = 0 -> start pose
        alpha = 1 -> end pose
        """

        translation = (
            (1.0 - alpha) * translation_start
            + alpha * translation_end
        )

        rotations = Rotation.from_matrix([
            rotation_start,
            rotation_end
        ])

        slerp = Slerp(
            [0.0, 1.0],
            rotations
        )

        rotation = slerp([alpha])[0]

        return translation, rotation

    # --------------------------------------------------------
    # Deskew
    # --------------------------------------------------------

    def deskew(
        self,
        points,
        translation_start,
        translation_end,
        rotation_start,
        rotation_end
    ):
        """
        Deskew CARLA LiDAR point cloud.

        Input:
            points:
                Nx4 [x, y, z, intensity]

            translation_start:
                Vehicle position at scan start.

            translation_end:
                Vehicle position at scan end.

            rotation_start:
                Vehicle rotation at scan start.

            rotation_end:
                Vehicle rotation at scan end.

        Output:
            Nx4 deskewed point cloud.
        """

        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                "Expected LiDAR points with shape (N, 4)"
            )

        num_points = points.shape[0]

        if num_points == 0:
            return points.copy()

        # Approximate acquisition time
        point_times = self.create_point_times(
            num_points
        )

        alphas = (
            point_times /
            self.scan_duration
        )

        deskewed_points = np.zeros_like(
            points,
            dtype=np.float64
        )

        # Scan-start pose is the reference
        R_start = rotation_start
        t_start = translation_start

        for i in range(num_points):

            alpha = alphas[i]

            t_point, R_point = self.interpolate_pose(
                alpha,
                translation_start,
                translation_end,
                rotation_start,
                rotation_end
            )

            R_point_matrix = R_point.as_matrix()

            # Original LiDAR point
            p = points[i, :3]

            # Preserve intensity
            intensity = points[i, 3]

            # LiDAR frame -> world
            p_world = (
                R_point_matrix @ p
                + t_point
            )

            # World -> scan-start frame
            p_reference = (
                R_start.T @
                (p_world - t_start)
            )

            deskewed_points[i, :3] = p_reference
            deskewed_points[i, 3] = intensity

        return deskewed_points
