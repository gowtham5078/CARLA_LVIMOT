import sys
import numpy as np

sys.path.insert(0, "src")

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_lidar_features import CarlaLiDARPlanarFeatureExtractor
from carla_box_projection import CarlaBoundingBoxProjector
from carla_association import CarlaObjectAssociator
from carla_multimodal_features import CarlaMultimodalFeatureBuilder


CARLA_ROOT = "~/catkin_ws/src/CARLA_LVIMOT"
SEQUENCE = "0006"
FRAME = 100

reader = CarlaReader(
    CARLA_ROOT,
    SEQUENCE
)

calibration = CarlaCalibration(
    f"{CARLA_ROOT}/calibration/carla_calibration.json"
)

# ---------------------------------------------------------
# Load frame data
# ---------------------------------------------------------

lidar = reader.load_lidar(FRAME)
image = reader.load_image(FRAME)
pose = reader.load_pose(FRAME)

objects = reader.load_labels(FRAME)["objects"]

print("=" * 70)
print("CARLA MULTIMODAL FEATURE TEST")
print("=" * 70)

print(f"Frame: {FRAME}")
print(f"LiDAR points: {len(lidar)}")
print(f"Image shape: {image.shape}")
print(f"Objects: {len(objects)}")

# ---------------------------------------------------------
# LiDAR planar features
# ---------------------------------------------------------

feature_extractor = CarlaLiDARPlanarFeatureExtractor()

planar_features = feature_extractor.extract(
    lidar[:, :3]
)

print(
    f"Planar features: {len(planar_features)}"
)

# ---------------------------------------------------------
# Project 3D boxes
# ---------------------------------------------------------

detector = CarlaDetector(
    reader,
    calibration
)

detections = detector.load_detections(
    FRAME
)

box_projector = CarlaBoundingBoxProjector(
    calibration
)

projected_boxes = {}

for obj in detections:

    actor_id = int(
        obj["actor_id"]
    )

    world_corners = np.asarray(
        obj["bounding_box"]["world_corners"],
        dtype=np.float64
    )

    projected, valid, bbox = (
        box_projector.project_world_corners(
            world_corners,
            pose,
            image.shape[:2]
        )
    )

    if bbox is not None:

        projected_boxes[actor_id] = {
            "bbox": bbox,
            "projected_corners": projected,
            "valid": valid
        }

# ---------------------------------------------------------
# Camera objects
#
# For this validation, use the projected CARLA GT boxes.
# This validates the multimodal builder itself.
# ---------------------------------------------------------

camera_objects = []

for obj in detections:

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

# ---------------------------------------------------------
# Association
# ---------------------------------------------------------

associator = CarlaObjectAssociator(
    iou_threshold=0.1
)

associations = associator.associate(
    detections,
    camera_objects,
    projected_boxes
)

print(
    f"Projected objects: "
    f"{len(projected_boxes)}"
)

print(
    f"Associations: "
    f"{len(associations)}"
)

# ---------------------------------------------------------
# Dummy camera features
#
# We first test LiDAR multimodal integration.
# Real FAST/BRIEF will be connected separately.
# ---------------------------------------------------------

keypoints = np.empty(
    (0, 2),
    dtype=np.float64
)

descriptors = np.empty(
    (0, 32),
    dtype=np.uint8
)

# ---------------------------------------------------------
# Build multimodal features
# ---------------------------------------------------------

builder = CarlaMultimodalFeatureBuilder(
    calibration
)

multimodal_objects = builder.build(
    objects=detections,
    associations=associations,
    projected_boxes=projected_boxes,
    planar_features=planar_features,
    ego_pose=pose,
    keypoints=keypoints,
    descriptors=descriptors
)

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 70)
print("MULTIMODAL OBJECT RESULTS")
print("=" * 70)

total_lidar_features = 0

for obj in multimodal_objects:

    lidar_count = (
        obj["lidar_features"]["count"]
    )

    camera_count = (
        obj["camera_features"]["count"]
    )

    total_lidar_features += lidar_count

    print(
        f"Actor {obj['actor_id']:3d} | "
        f"{obj['type']:<45} | "
        f"LiDAR: {lidar_count:4d} | "
        f"Camera: {camera_count:4d}"
    )

print()
print(
    f"Multimodal objects: "
    f"{len(multimodal_objects)}"
)

print(
    f"Total LiDAR features: "
    f"{total_lidar_features}"
)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(multimodal_objects) == len(
    associations
)

for obj in multimodal_objects:

    lf = obj["lidar_features"]

    assert lf["points"].ndim == 2
    assert lf["points"].shape[1] == 3

    assert lf["normals"].ndim == 2
    assert lf["normals"].shape[1] == 3

    assert len(lf["points"]) == lf["count"]
    assert len(lf["normals"]) == lf["count"]

    assert np.isfinite(
        lf["points"]
    ).all()

    assert np.isfinite(
        lf["normals"]
    ).all()

print()
print("=" * 70)
print("CARLA MULTIMODAL FEATURE TEST PASSED")
print("=" * 70)
