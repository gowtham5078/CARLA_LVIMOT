import sys
import numpy as np

sys.path.insert(0, "src")

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_box_projection import CarlaBoundingBoxProjector
from carla_association import CarlaObjectAssociator
from carla_multimodal_features import CarlaMultimodalFeatureBuilder
from fast_features import FASTFeatureDetector
from brief_features import BRIEFDescriptorExtractor


CARLA_ROOT = "~/catkin_ws/src/CARLA_LVIMOT"
SEQUENCE = "0006"
FRAME = 100


# ---------------------------------------------------------
# Load CARLA data
# ---------------------------------------------------------

reader = CarlaReader(
    CARLA_ROOT,
    SEQUENCE
)

calibration = CarlaCalibration(
    f"{CARLA_ROOT}/calibration/carla_calibration.json"
)

image = reader.load_image(FRAME)
pose = reader.load_pose(FRAME)

detector = CarlaDetector(
    reader,
    calibration
)

objects = detector.load_detections(FRAME)


print("=" * 70)
print("CARLA CAMERA FEATURE TEST")
print("=" * 70)

print(f"Frame: {FRAME}")
print(f"Image shape: {image.shape}")
print(f"Objects: {len(objects)}")


# ---------------------------------------------------------
# FAST
# ---------------------------------------------------------

fast_detector = FASTFeatureDetector(
    threshold=20,
    nonmax_suppression=True
)

fast_keypoints = fast_detector.detect(
    image
)

print()
print(f"FAST keypoints: {len(fast_keypoints)}")


# ---------------------------------------------------------
# BRIEF
# ---------------------------------------------------------

brief_extractor = BRIEFDescriptorExtractor(
    bytes=32
)

brief_keypoints, descriptors = (
    brief_extractor.compute(
        image,
        fast_keypoints
    )
)

if descriptors is None:
    descriptors = np.empty(
        (0, 32),
        dtype=np.uint8
    )

print(
    f"BRIEF keypoints: "
    f"{len(brief_keypoints)}"
)

print(
    f"BRIEF descriptors: "
    f"{descriptors.shape}"
)


# ---------------------------------------------------------
# Convert keypoints to (x, y)
# ---------------------------------------------------------

keypoints_xy = np.asarray(
    [
        [kp.pt[0], kp.pt[1]]
        for kp in brief_keypoints
    ],
    dtype=np.float64
)

print(
    f"Keypoint array: "
    f"{keypoints_xy.shape}"
)


# ---------------------------------------------------------
# Project CARLA 3D boxes
# ---------------------------------------------------------

box_projector = CarlaBoundingBoxProjector(
    calibration
)

projected_boxes = {}

for obj in objects:

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
# For this validation, projected CARLA GT boxes
# represent the camera detections.
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Association
# ---------------------------------------------------------

associator = CarlaObjectAssociator(
    iou_threshold=0.1
)

associations = associator.associate(
    objects,
    camera_objects,
    projected_boxes
)

print()
print(
    f"Projected objects: "
    f"{len(projected_boxes)}"
)

print(
    f"Associations: "
    f"{len(associations)}"
)


# ---------------------------------------------------------
# Build multimodal camera features
#
# We don't need LiDAR features for this test.
# ---------------------------------------------------------

builder = CarlaMultimodalFeatureBuilder(
    calibration
)

empty_planar_features = []

multimodal_objects = builder.build(
    objects=objects,
    associations=associations,
    projected_boxes=projected_boxes,
    planar_features=empty_planar_features,
    ego_pose=pose,
    keypoints=keypoints_xy,
    descriptors=descriptors
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print()
print("=" * 70)
print("CAMERA FEATURES INSIDE OBJECT BOXES")
print("=" * 70)

total_camera_features = 0

for obj in multimodal_objects:

    camera_features = (
        obj["camera_features"]
    )

    count = camera_features["count"]

    total_camera_features += count

    print(
        f"Actor {obj['actor_id']:3d} | "
        f"{obj['type']:<45} | "
        f"Camera features: {count:4d}"
    )

    print(
        f"    keypoints: "
        f"{camera_features['keypoints'].shape}"
    )

    print(
        f"    descriptors: "
        f"{camera_features['descriptors'].shape}"
    )


print()
print(
    f"Multimodal objects: "
    f"{len(multimodal_objects)}"
)

print(
    f"Total camera features: "
    f"{total_camera_features}"
)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(fast_keypoints) > 0
assert len(brief_keypoints) > 0
assert descriptors.ndim == 2
assert descriptors.shape[1] == 32

assert len(keypoints_xy) == len(descriptors)

assert len(multimodal_objects) == len(
    associations
)

for obj in multimodal_objects:

    camera_features = (
        obj["camera_features"]
    )

    assert camera_features["keypoints"].ndim == 2
    assert camera_features["keypoints"].shape[1] == 2

    assert camera_features["descriptors"].ndim == 2
    assert camera_features["descriptors"].shape[1] == 32

    assert (
        len(camera_features["keypoints"])
        ==
        len(camera_features["descriptors"])
    )

print()
print("=" * 70)
print("CARLA CAMERA FEATURE TEST PASSED")
print("=" * 70)
