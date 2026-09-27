# CARLA-LVIMOT

## LiDAR-Visual-Inertial Fusion for Localization and Multi-Object Tracking

CARLA-based implementation of the LVIMOT pipeline for autonomous driving.

The project combines LiDAR, camera and IMU information for multimodal perception and temporal multi-object tracking in dynamic environments.

## Current Implementation

### Sensor and perception front-end

- CARLA dataset reader
- Camera calibration
- LiDAR processing
- IMU processing
- IMU preintegration
- LiDAR deskewing
- LiDAR geometric feature extraction
- FAST feature extraction
- BRIEF descriptors
- LiDAR-camera projection
- 3D bounding-box projection
- Object association
- Multimodal feature generation

### Temporal tracking

A baseline multi-object tracker is implemented using:

- Constant-velocity motion model
- 6D state [x, y, z, vx, vy, vz]
- Kalman prediction
- Kalman measurement update
- Class-constrained association
- Hungarian assignment
- Euclidean distance gating
- Track creation
- Missed-track handling
- Track deletion
- Track history

## Dataset

The CARLA dataset contains multiple sequences with RGB camera, 64-channel LiDAR, IMU, ego pose, and object ground truth.

The dataset is generated separately and is not included in this repository.

## Project Status

Completed:

- CARLA dataset generation
- Sensor synchronization
- Calibration
- IMU processing
- IMU preintegration
- LiDAR deskewing
- LiDAR feature extraction
- FAST + BRIEF
- LiDAR-camera projection
- 3D box projection
- Object association
- Multimodal feature generation
- 500-frame CARLA pipeline execution
- Baseline temporal multi-object tracking
- Initial tracking evaluation

In progress:

- Velocity evaluation
- Dynamic/static classification
- Temporal LiDAR tracking
- Temporal visual tracking
- Ego-motion-aware tracking
- Factor graph optimization
- Ceres optimization
- Sliding-window estimation
- 4D mapping
- Quantitative localization evaluation

## Repository Structure

```text
CARLA_LVIMOT/
├── config/
├── src/
└── tests/
```

## Important Note

The current CARLA object observations are generated from CARLA ground-truth actor information. They are used as controlled observations for development and evaluation of the temporal tracking pipeline.

The current implementation is a research/development baseline and is not intended to represent the complete original LVIMOT algorithm.

## Environment

Tested with Python 3, CARLA, NumPy, SciPy, OpenCV, and Open3D.
