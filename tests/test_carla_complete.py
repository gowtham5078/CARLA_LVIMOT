import sys
import numpy as np

sys.path.insert(0, "src")

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_box_projection import CarlaBoundingBoxProjector
from carla_association import CarlaObjectAssociator
from carla_lidar_features import CarlaLiDARPlanarFeatureExtractor
from carla_multimodal_features import CarlaMultimodalFeatureBuilder
from fast_features import FASTFeatureDetector
from brief_features import BRIEFDescriptorExtractor


CARLA_ROOT = "../CARLA_LVIMOT"
SEQUENCE = "0006"
FRAME = 100

reader = CarlaReader(CARLA_ROOT, SEQUENCE)
calibration = CarlaCalibration(
    f"{CARLA_ROOT}/calibration/carla_calibration.json"
)

detector = CarlaDetector(reader, calibration)
projector = CarlaBoundingBoxProjector(calibration)
associator = CarlaObjectAssociator(iou_threshold=0.1)

print("\n========== COMPLETE CARLA MULTIMODAL TEST ==========\n")

# ----------------------------------------------------
# 1. Load data
# ----------------------------------------------------
image = reader.load_image(FRAME)
lidar = reader.load_lidar(FRAME)
ego_pose = reader.load_pose(FRAME)
objects = detector.load_detections(FRAME)

print(f"Frame: {FRAME}")
print(f"Image shape: {image.shape}")
print(f"LiDAR points: {len(lidar)}")
print(f"Objects: {len(objects)}")

# ----------------------------------------------------
# 2. LiDAR planar features
# ----------------------------------------------------
lidar_extractor = CarlaLiDARPlanarFeatureExtractor()

planar_features = lidar_extractor.extract(lidar)

print(f"Planar features: {len(planar_features)}")

# ----------------------------------------------------
# 3. FAST + BRIEF
# ----------------------------------------------------
fast = FASTFeatureDetector(
    threshold=20,
    nonmax_suppression=True
)

brief = BRIEFDescriptorExtractor(bytes=32)

fast_keypoints = fast.detect(image)

brief_keypoints, descriptors = brief.compute(
    image,
    fast_keypoints
)

print(f"FAST keypoints: {len(fast_keypoints)}")
print(f"BRIEF keypoints: {len(brief_keypoints)}")
print(f"BRIEF descriptors: {descriptors.shape}")

assert descriptors is not None
assert descriptors.shape[1] == 32
assert len(brief_keypoints) == descriptors.shape[0]

# ----------------------------------------------------
# 4. Project 3D object boxes
# ----------------------------------------------------
projected_boxes = {}

for obj in objects:

    corners = np.asarray(
        obj["bounding_box"]["world_corners"],
        dtype=np.float64
    )

    projected, valid, bbox = projector.project_world_corners(
        corners,
        ego_pose,
        image.shape
    )

    if bbox is not None:
        projected_boxes[obj["actor_id"]] = {
            "bbox": bbox,
            "projected_corners": projected,
            "valid": valid
        }

print(f"Projected objects: {len(projected_boxes)}")

# ----------------------------------------------------
# 5. Create validation camera objects
# ----------------------------------------------------
camera_objects = []

for obj in objects:

    actor_id = obj["actor_id"]

    if actor_id not in projected_boxes:
        continue

    camera_objects.append({
        "actor_id": actor_id,
        "type": obj["type"],
        "bbox": projected_boxes[actor_id]["bbox"]
    })

print(f"Camera objects: {len(camera_objects)}")

# ----------------------------------------------------
# 6. Association
# ----------------------------------------------------
associations = associator.associate(
    objects,
    camera_objects,
    projected_boxes
)

print(f"Associations: {len(associations)}")

for a in associations:
    print(
        f"  3D Actor {a['lidar_id']} -> "
        f"Camera Actor {a['camera_id']} | "
        f"{a['class']} | "
        f"IoU={a['iou']:.6f}"
    )

# ----------------------------------------------------
# 7. Build multimodal features
# ----------------------------------------------------
builder = CarlaMultimodalFeatureBuilder(
    calibration,
    max_lidar_distance=None
)

multimodal_objects = builder.build(
    objects=objects,
    associations=associations,
    projected_boxes=projected_boxes,
    planar_features=planar_features,
    ego_pose=ego_pose,
    keypoints=brief_keypoints,
    descriptors=descriptors
)

print(f"\nMultimodal objects: {len(multimodal_objects)}")

# ----------------------------------------------------
# 8. Print results
# ----------------------------------------------------
total_lidar_features = 0
total_camera_features = 0

for obj in multimodal_objects:

    lidar_count = len(obj["lidar_features"])
    camera_count = len(obj["camera_features"])

    total_lidar_features += lidar_count
    total_camera_features += camera_count

    print(
        f"\nActor {obj['actor_id']} | "
        f"{obj['type']}"
    )

    print(f"  LiDAR features: {lidar_count}")
    print(f"  Camera features: {camera_count}")

    if camera_count > 0:
        kp = np.asarray(
            obj["camera_features"][0]["keypoint"]
        )

        desc = np.asarray(
            obj["camera_features"][0]["descriptor"]
        )

        print(f"  Example keypoint: {kp}")
        print(f"  Example descriptor shape: {desc.shape}")

print("\n========== SUMMARY ==========")
print(f"LiDAR points:             {len(lidar)}")
print(f"Planar features:          {len(planar_features)}")
print(f"FAST keypoints:            {len(fast_keypoints)}")
print(f"BRIEF descriptors:         {descriptors.shape}")
print(f"Projected objects:        {len(projected_boxes)}")
print(f"Associations:             {len(associations)}")
print(f"Multimodal objects:       {len(multimodal_objects)}")
print(f"Total LiDAR features:     {total_lidar_features}")
print(f"Total camera features:    {total_camera_features}")

# ----------------------------------------------------
# 9. Basic validation
# ----------------------------------------------------
assert lidar.ndim == 2
assert lidar.shape[1] == 4
assert np.isfinite(lidar[:, :3]).all()

assert len(brief_keypoints) == descriptors.shape[0]
assert descriptors.shape[1] == 32

assert len(multimodal_objects) == 1

assert total_camera_features == 94

print("\n==============================================")
print("COMPLETE CARLA MULTIMODAL FRAME TEST PASSED")
print("==============================================")
