import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_box_projection import CarlaBoundingBoxProjector
from carla_association import CarlaObjectAssociator
from carla_lidar_features import CarlaLiDARPlanarFeatureExtractor
from carla_multimodal_features import CarlaMultimodalFeatureBuilder
from carla_lidar_deskew import CarlaLiDARDeskewer
from carla_imu import CarlaIMUPreprocessor
from carla_imu_motion import CarlaIMUMotion
from imu_preintegration import IMUPreintegrator
from fast_features import FASTFeatureDetector
from brief_features import BRIEFDescriptorExtractor


class CarlaLVIMOTPipeline:

    def __init__(
        self,
        carla_root,
        sequence,
        output_root="outputs_carla",
        brief_bytes=32,
        voxel_size=0.20
    ):

        self.carla_root = os.path.expanduser(carla_root)
        self.sequence = str(sequence).zfill(4)

        self.output_root = os.path.expanduser(output_root)

        self.reader = CarlaReader(
            self.carla_root,
            self.sequence
        )

        calibration_file = os.path.join(
            self.carla_root,
            "calibration",
            "carla_calibration.json"
        )

        self.calibration = CarlaCalibration(
            calibration_file
        )

        # -----------------------------------------
        # CARLA modules
        # -----------------------------------------

        self.detector = CarlaDetector(
            self.reader,
            self.calibration
        )

        self.box_projector = CarlaBoundingBoxProjector(
            self.calibration
        )

        self.associator = CarlaObjectAssociator(
            iou_threshold=0.1
        )

        self.lidar_features = CarlaLiDARPlanarFeatureExtractor(
            voxel_size=voxel_size
        )

        self.multimodal_builder = CarlaMultimodalFeatureBuilder(
            self.calibration
        )

        self.lidar_deskewer = CarlaLiDARDeskewer(
            scan_duration=0.1
        )

        self.imu_preprocessor = CarlaIMUPreprocessor(
            dt=0.1
        )

        self.imu_motion = CarlaIMUMotion(
            self.reader
        )

        self.imu_preintegrator = IMUPreintegrator()

        self.fast = FASTFeatureDetector(
            threshold=20,
            nonmax_suppression=True
        )

        self.brief = BRIEFDescriptorExtractor(
            bytes=brief_bytes
        )

        self.output_sequence = os.path.join(
            self.output_root,
            f"sequence_{self.sequence}"
        )

        os.makedirs(
            self.output_sequence,
            exist_ok=True
        )

        self.output_file = os.path.join(
            self.output_sequence,
            "results.json"
        )


    # =========================================================
    # Frame processing
    # =========================================================

    def process_frame(self, frame):

        print("\n" + "=" * 65)
        print(f"Processing CARLA sequence {self.sequence}, frame {frame}")
        print("=" * 65)

        # -----------------------------------------------------
        # 1. Load CARLA data
        # -----------------------------------------------------

        image = self.reader.load_image(frame)

        lidar = self.reader.load_lidar(frame)

        imu_data = self.reader.load_imu(frame)

        ego_pose = self.reader.load_pose(frame)

        objects = self.detector.load_detections(frame)

        print(f"[1/10] Image: {image.shape}")
        print(f"[1/10] LiDAR points: {len(lidar)}")
        print(f"[1/10] Objects: {len(objects)}")


        # -----------------------------------------------------
        # 2. IMU preprocessing
        # -----------------------------------------------------

        imu_motion = self.imu_preprocessor.process(
            imu_data
        )

        print(
            "[2/10] IMU acceleration:",
            imu_motion["acceleration"]
        )

        print(
            "[2/10] IMU gyro:",
            imu_motion["gyro"]
        )


        # -----------------------------------------------------
        # 3. IMU preintegration
        # -----------------------------------------------------

        imu_preintegrated = None

        if frame > 0:

            previous_imu = self.reader.load_imu(
                frame - 1
            )

            previous_motion = self.imu_preprocessor.process(
                previous_imu
            )

            imu_preintegrated = self.imu_preintegrator.integrate(
                previous_motion,
                imu_motion
            )

        print("[3/10] IMU preintegration complete")


        # -----------------------------------------------------
        # 4. LiDAR deskew
        # -----------------------------------------------------

        deskewed_lidar = lidar

        if frame > 0:

            try:

                previous_pose = self.reader.load_pose(
                    frame - 1
                )

                # ---------------------------------------------
                # Extract scan-start and scan-end translations
                # ---------------------------------------------

                translation_start = np.array([
                    previous_pose["location"]["x"],
                    previous_pose["location"]["y"],
                    previous_pose["location"]["z"]
                ], dtype=np.float64)

                translation_end = np.array([
                    ego_pose["location"]["x"],
                    ego_pose["location"]["y"],
                    ego_pose["location"]["z"]
                ], dtype=np.float64)

                # ---------------------------------------------
                # Extract scan-start and scan-end rotations
                #
                # The deskewer expects 3x3 rotation matrices,
                # not the raw CARLA rotation dictionaries.
                # ---------------------------------------------

                rotation_start = self.imu_motion.rotation_matrix(
                    previous_pose["rotation"]
                )

                rotation_end = self.imu_motion.rotation_matrix(
                    ego_pose["rotation"]
                )

                # ---------------------------------------------
                # Perform approximate LiDAR deskewing
                # ---------------------------------------------

                deskewed_lidar = self.lidar_deskewer.deskew(
                    lidar,
                    translation_start,
                    translation_end,
                    rotation_start,
                    rotation_end
                )

            except Exception as e:

                print(
                    "[4/10] Deskew warning:",
                    str(e)
                )

                deskewed_lidar = lidar

        print(
            f"[4/10] Deskewed LiDAR points: "
            f"{len(deskewed_lidar)}"
        )


        # -----------------------------------------------------
        # 5. LiDAR planar features
        # -----------------------------------------------------

        planar_features = self.lidar_features.extract(
            deskewed_lidar
        )

        print(
            f"[5/10] Planar features: "
            f"{len(planar_features)}"
        )


        # -----------------------------------------------------
        # 6. FAST + BRIEF
        # -----------------------------------------------------

        fast_keypoints = self.fast.detect(
            image
        )

        brief_keypoints, descriptors = self.brief.compute(
            image,
            fast_keypoints
        )

        if descriptors is None:

            descriptors = np.empty(
                (0, 32),
                dtype=np.uint8
            )

        print(
            f"[6/10] FAST keypoints: "
            f"{len(fast_keypoints)}"
        )

        print(
            f"[6/10] BRIEF descriptors: "
            f"{descriptors.shape}"
        )


        # -----------------------------------------------------
        # 7. Project 3D boxes into camera
        # -----------------------------------------------------

        projected_boxes = {}

        for obj in objects:

            actor_id = int(
                obj["actor_id"]
            )

            corners = np.asarray(
                obj["bounding_box"]["world_corners"],
                dtype=np.float64
            )

            projected, valid, bbox = (
                self.box_projector.project_world_corners(
                    corners,
                    ego_pose,
                    image.shape
                )
            )

            if bbox is not None:

                projected_boxes[actor_id] = {
                    "bbox": bbox,
                    "projected_corners": projected,
                    "valid": valid
                }

        print(
            f"[7/10] Projected objects: "
            f"{len(projected_boxes)}"
        )


        # -----------------------------------------------------
        # 8. Camera objects for CARLA validation
        # -----------------------------------------------------
        #
        # IMPORTANT:
        # These are projected CARLA ground-truth boxes.
        # They are NOT an independent camera detector.
        #

        camera_objects = []

        for obj in objects:

            actor_id = int(
                obj["actor_id"]
            )

            if actor_id not in projected_boxes:
                continue

            camera_objects.append({
                "actor_id": actor_id,
                "type": obj["type"],
                "bbox": projected_boxes[
                    actor_id
                ]["bbox"]
            })

        print(
            f"[8/10] Camera objects: "
            f"{len(camera_objects)}"
        )


        # -----------------------------------------------------
        # 9. Association
        # -----------------------------------------------------

        associations = self.associator.associate(
            objects,
            camera_objects,
            projected_boxes
        )

        print(
            f"[9/10] Associations: "
            f"{len(associations)}"
        )


        # -----------------------------------------------------
        # 10. Multimodal features
        # -----------------------------------------------------

        multimodal_objects = (
            self.multimodal_builder.build(
                objects=objects,
                associations=associations,
                projected_boxes=projected_boxes,
                planar_features=planar_features,
                ego_pose=ego_pose,
                keypoints=brief_keypoints,
                descriptors=descriptors
            )
        )

        print(
            f"[10/10] Multimodal objects: "
            f"{len(multimodal_objects)}"
        )


        # -----------------------------------------------------
        # Convert NumPy values to JSON-safe values
        # -----------------------------------------------------

        def json_safe(value):

            if isinstance(value, np.ndarray):
                return value.tolist()

            if isinstance(value, np.integer):
                return int(value)

            if isinstance(value, np.floating):
                return float(value)

            if isinstance(value, dict):

                return {
                    str(k): json_safe(v)
                    for k, v in value.items()
                }

            if isinstance(value, list):

                return [
                    json_safe(v)
                    for v in value
                ]

            if isinstance(value, tuple):

                return [
                    json_safe(v)
                    for v in value
                ]

            return value


        # -----------------------------------------------------
        # Frame result
        # -----------------------------------------------------

        result = {

            "sequence": self.sequence,

            "frame": int(frame),

            "timestamp": float(
                self.reader.load_timestamp(frame)
            ),

            "image": {
                "height": int(image.shape[0]),
                "width": int(image.shape[1])
            },

            "lidar": {
                "raw_points": int(len(lidar)),
                "deskewed_points": int(
                    len(deskewed_lidar)
                ),
                "planar_features": int(
                    len(planar_features)
                )
            },

            "imu": {
                "motion": imu_motion,
                "preintegration": imu_preintegrated
            },

            "camera": {
                "fast_keypoints": int(
                    len(fast_keypoints)
                ),

                "brief_keypoints": int(
                    len(brief_keypoints)
                ),

                "descriptor_shape": list(
                    descriptors.shape
                )
            },

            "objects": {
                "ground_truth": int(
                    len(objects)
                ),

                "projected": int(
                    len(projected_boxes)
                ),

                "camera_objects": int(
                    len(camera_objects)
                ),

                "associations": associations
            },

            "multimodal_objects":
                multimodal_objects
        }

        return json_safe(result)


    # =========================================================
    # Save frame result
    # =========================================================

    def save_frame(self, result):

        frame = int(
            result["frame"]
        )

        frame_file = os.path.join(
            self.output_sequence,
            f"frame_{frame:06d}.json"
        )

        with open(
            frame_file,
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

        return frame_file


    # =========================================================
    # Run sequence
    # =========================================================

    def run(
        self,
        start_frame=0,
        end_frame=None
    ):

        # Determine number of frames from timestamps
        timestamp_dir = os.path.join(
            self.carla_root,
            "sequences",
            self.sequence
        )

        timestamp_file = os.path.join(
            timestamp_dir,
            "timestamps.txt"
        )

        with open(timestamp_file) as f:

            total_frames = sum(
                1
                for line in f
                if line.strip()
            )

        if end_frame is None:

            end_frame = total_frames - 1

        end_frame = min(
            end_frame,
            total_frames - 1
        )

        print("\n")
        print("=" * 70)
        print("CARLA LVIMOT PIPELINE")
        print("=" * 70)
        print(f"Sequence:    {self.sequence}")
        print(f"Start frame: {start_frame}")
        print(f"End frame:   {end_frame}")
        print(f"Total frames: {total_frames}")
        print("=" * 70)

        results = []

        for frame in range(
            start_frame,
            end_frame + 1
        ):

            try:

                result = self.process_frame(
                    frame
                )

                frame_file = self.save_frame(
                    result
                )

                results.append(result)

                print(
                    f"Saved: {frame_file}"
                )

            except Exception as e:

                print(
                    "\nERROR processing frame "
                    f"{frame}: {e}"
                )

                raise

        # -----------------------------------------------------
        # Sequence summary
        # -----------------------------------------------------

        summary = {

            "sequence": self.sequence,

            "start_frame": int(start_frame),

            "end_frame": int(end_frame),

            "processed_frames": int(
                len(results)
            ),

            "frames": results
        }

        with open(
            self.output_file,
            "w"
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

        print("\n")
        print("=" * 70)
        print("SEQUENCE PROCESSING COMPLETE")
        print("=" * 70)
        print(
            f"Processed frames: {len(results)}"
        )
        print(
            f"Results directory: "
            f"{self.output_sequence}"
        )
        print(
            f"Sequence JSON: "
            f"{self.output_file}"
        )
        print("=" * 70)


def main():

    parser = argparse.ArgumentParser(
        description="CARLA LVIMOT Python Pipeline"
    )

    parser.add_argument(
        "--carla-root",
        default="~/catkin_ws/src/CARLA_LVIMOT"
    )

    parser.add_argument(
        "--sequence",
        required=True
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None
    )

    parser.add_argument(
        "--output-root",
        default="outputs_carla"
    )

    args = parser.parse_args()

    pipeline = CarlaLVIMOTPipeline(
        carla_root=args.carla_root,
        sequence=args.sequence,
        output_root=args.output_root
    )

    pipeline.run(
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )


if __name__ == "__main__":
    main()
