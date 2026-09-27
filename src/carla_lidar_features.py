import numpy as np
import open3d as o3d


class CarlaLiDARPlanarFeatureExtractor:

    def __init__(
        self,
        voxel_size=0.20,
        search_radius=0.50,
        min_neighbors=10,
        planarity_threshold=0.15
    ):
        self.voxel_size = voxel_size
        self.search_radius = search_radius
        self.min_neighbors = min_neighbors
        self.planarity_threshold = planarity_threshold

    # --------------------------------------------------
    # Voxel downsampling
    # --------------------------------------------------

    def voxel_downsample(self, points):

        points = np.asarray(
            points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Expected point cloud with shape (N, >=3)"
            )

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            points[:, :3]
        )

        downsampled = pcd.voxel_down_sample(
            voxel_size=self.voxel_size
        )

        return np.asarray(
            downsampled.points
        )

    # --------------------------------------------------
    # Extract planar features
    # --------------------------------------------------

    def extract(self, points):

        points = np.asarray(
            points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Expected point cloud with shape (N, >=3)"
            )

        # ----------------------------------------------
        # 1. Remove invalid points
        # ----------------------------------------------

        valid = np.isfinite(
            points[:, :3]
        ).all(axis=1)

        points = points[valid]

        if len(points) == 0:
            return []

        # ----------------------------------------------
        # 2. Downsample
        # ----------------------------------------------

        points = self.voxel_downsample(
            points
        )

        print(
            f"Points after downsampling: {len(points)}"
        )

        if len(points) == 0:
            return []

        # ----------------------------------------------
        # 3. Create Open3D point cloud
        # ----------------------------------------------

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            points
        )

        # ----------------------------------------------
        # 4. KD-tree
        # ----------------------------------------------

        kdtree = o3d.geometry.KDTreeFlann(
            pcd
        )

        features = []

        # ----------------------------------------------
        # 5. Analyze every point
        # ----------------------------------------------

        for i, point in enumerate(points):

            count, indices, distances = (
                kdtree.search_radius_vector_3d(
                    point,
                    self.search_radius
                )
            )

            if count < self.min_neighbors:
                continue

            neighborhood = points[
                indices
            ]

            # ------------------------------------------
            # 6. Neighborhood centroid
            # ------------------------------------------

            centroid = np.mean(
                neighborhood,
                axis=0
            )

            centered = (
                neighborhood -
                centroid
            )

            # ------------------------------------------
            # 7. Covariance matrix
            # ------------------------------------------

            covariance = (
                centered.T @ centered
            ) / max(
                len(neighborhood) - 1,
                1
            )

            # ------------------------------------------
            # 8. Eigenvalue decomposition
            # ------------------------------------------

            eigenvalues, eigenvectors = np.linalg.eigh(
                covariance
            )

            order = np.argsort(
                eigenvalues
            )

            eigenvalues = eigenvalues[
                order
            ]

            eigenvectors = eigenvectors[
                :, order
            ]

            lambda0 = eigenvalues[0]
            lambda1 = eigenvalues[1]
            lambda2 = eigenvalues[2]

            total = (
                lambda0 +
                lambda1 +
                lambda2
            )

            if total <= 1e-12:
                continue

            # ------------------------------------------
            # 9. Surface variation
            # ------------------------------------------

            surface_variation = (
                lambda0 / total
            )

            # ------------------------------------------
            # 10. Planarity
            # ------------------------------------------

            planarity = (
                (lambda1 - lambda0)
                /
                max(lambda2, 1e-12)
            )

            # ------------------------------------------
            # 11. Select planar feature
            # ------------------------------------------

            if planarity >= self.planarity_threshold:

                normal = eigenvectors[:, 0]

                normal = (
                    normal /
                    max(
                        np.linalg.norm(normal),
                        1e-12
                    )
                )

                features.append({
                    "point": point,
                    "normal": normal,
                    "eigenvalues": eigenvalues,
                    "surface_variation":
                        surface_variation,
                    "planarity":
                        planarity
                })

        return features
