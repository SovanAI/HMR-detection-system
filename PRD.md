# Product Requirements Document (PRD)
## BAS-HMR: Real-Time Person–Chair Distance System

**Project Name:** BAS-HMR  
**Version:** 2.0  
**Date Updated:** 2026-09-15  
**Status:** Active Development / Real-time ML distance pipeline  

---

## 1. Executive Summary

BAS-HMR is a real-time computer vision system for estimating the distance between a person and a chair using live camera input. The current codebase is not a full HMR/SMPL mesh reconstruction system; it is a working perception pipeline that combines object detection, metric depth estimation, 3D grid feature extraction, and a trained regression model to predict spatial distance in meters.

The active implementation centers on these components:
- OpenCV camera capture and MJPEG streaming
- Ultralytics YOLO11n for person/chair detection
- YOLO26n depth model for metric depth maps
- 3D object grid extraction from depth + bounding boxes
- Pairwise feature extraction for person–chair relations
- A trained Ridge regression model for distance prediction
- Flask endpoints for live visualization and telemetry

**Current Product Value Proposition:**
- Estimate person-to-chair distance in real time from a webcam
- Support multiple people and multiple chairs in the same scene
- Use independent trackers per object class to reduce identity confusion
- Reuse depth maps between frames to reduce runtime cost
- Visualize results via live MJPEG and JSON status APIs

---

## 2. Problem Statement

### Current Need
The project addresses a practical sensing task: understanding spatial relationship between a person and nearby seating objects in a live or recorded scene. This is useful for safety, room awareness, interaction analysis, and assistive monitoring.

### Challenges in the Current Implementation
- Person and chair detection must be class-specific and robust to overlaps
- Depth estimation is imperfect and must be filtered and validated
- 3D object features are more reliable than raw 2D boxes for distance estimation
- Real-time processing must keep camera stream stable without stalling on slow inference
- Multiple objects must be tracked independently without cross-class identity leakage

### Target Use Cases
1. Human-object proximity monitoring in room-scale scenes
2. Chair occupancy or person-seat distance analysis
3. Real-time validation of spatial interaction metrics
4. Dataset collection for ML distance model training
5. Live camera demos and debugging of distance estimation behavior

---

## 3. Goals and Objectives

### Primary Goals
1. Estimate person–chair distance from live camera input with acceptable real-time latency
2. Use a robust detection + depth + feature pipeline instead of relying on raw pixel distance alone
3. Support multiple objects while keeping per-class tracking independent
4. Make the system observable through a browser-friendly MJPEG stream and status endpoints
5. Train and deploy a compact regression model suitable for lightweight inference

### Success Criteria
- Live detection runs for person class 0 and chair class 56 only
- Distances are estimated in metric units from 3D object geometry
- Multiple detections are tracked without swapping person and chair identities
- The system continues functioning even when a depth or grid frame fails
- Output is accessible through both live visualization and structured API responses

---

## 4. Actual Technical Stack

### 4.1 Core Runtime
- Python 3.11.11
- OpenCV for camera acquisition, frame processing, and MJPEG output
- Flask for live web UI and status API
- NumPy for array processing and feature math
- joblib for loading the trained regression artifact
- scikit-learn Ridge regression model for final distance prediction

### 4.2 Detection and Depth Models
- Ultralytics YOLO11n
  - Used for person and chair detection
  - Restricted explicitly to COCO classes: person = 0, chair = 56
- Ultralytics YOLO26n-depth
  - Used for metric depth estimation from the input frame
  - Depth is reused between depth-refresh frames to maintain stability

### 4.3 3D Spatial Pipeline
- Camera intrinsics defined in a custom `CameraIntrinsics` structure
- `DepthTo3D` converts depth map values into 3D points
- `ObjectGridExtractor` builds a 5x5 grid over each detected object region
- `FeatureExtractor` computes per-object 3D statistics
- `PairFeatureExtractor` creates the final person–chair relationship vector
  - 34 feature order is treated as a contract for the distance model

### 4.4 Tracking and Smoothing
- `SimpleTracker` implements independent class-specific matching
- Per-class maximum center distance and missing-frame thresholds are configured
- Temporal smoothing is applied via EMA / rolling window logic before final output

### 4.5 Current Data and Model Artifacts
- Detection model: `yolo11n.pt`
- Depth model: `yolo26n-depth.pt`
- Regression model: `training/distance/models/best_distance_model.joblib`
- Output is displayed in a browser UI and also exposed through JSON endpoints

---

## 5. Current System Architecture

### Runtime Flow
1. Camera capture starts with OpenCV and a fixed resolution such as 640x480
2. YOLO11n detects only person and chair instances
3. Separate trackers maintain identity per class
4. A depth pass runs on a configured interval and refreshes the latest depth map
5. Bounding boxes are converted into local 3D object grids
6. Person and chair features are combined into a pairwise vector
7. The trained Ridge model predicts the metric distance
8. Temporal smoothing stabilizes the final value
9. The result is drawn on-frame and streamed to the web UI

### Important Implementation Principles
- YOLO11n is intentionally restricted to classes 0 and 56
- Person and chair tracking are independent, not shared
- Detections are drawn before depth/ML logic runs
- Grid extraction is performed from depth map + bounding box, not from raw dictionaries
- Failed depth or grid frames do not silently remove detections
- The latest valid depth map is reused between refresh intervals

---

## 6. Current Features and Behavior

### 6.1 Live Camera Processing
- The system supports live webcam input from a local camera source
- The processing loop captures frames at a fixed FPS target
- It handles multiple persons and chairs simultaneously
- Results are rendered on-screen with labels and overlays

### 6.2 Detection and Tracking
- Object classes are filtered to person and chair only
- `SimpleTracker` creates per-class identity assignments
- ID continuity is maintained using center-distance matching with a missing-frame allowance
- Detection and model stages are intentionally decoupled from tracking logic

### 6.3 Depth and 3D Geometry
- Metric depth is inferred from YOLO26n-depth output
- Depth map values are converted into 3D coordinates using camera intrinsics
- `Object3DGrid` stores the object’s spatial structure in a 5x5 grid layout
- Depth validity and sampling radius are used to reduce noise and outliers

### 6.4 ML Distance Estimation
- The final prediction uses a regressor trained on pairwise object features
- The feature vector contains person geometry, chair geometry, relative motion, depth relationships, and object distances
- The model is persisted as a joblib artifact and loaded at runtime

### 6.5 Visualization and API
- Live MJPEG stream is served through Flask
- A JSON status endpoint exposes runtime metadata
- The system is designed for local web inspection and iterative validation

---

## 7. Functional Requirements

### FR1: Camera Input
- The system SHALL accept live video streams from a local camera source
- The system SHALL support configurable capture width, height, and FPS
- The system SHALL allow a clean fallback when camera initialization fails

### FR2: Object Detection
- The system SHALL detect only person and chair classes in the active pipeline
- The system SHALL reject non-target classes before feature extraction
- The system SHALL support multiple detections in a single frame

### FR3: Tracking
- The system SHALL maintain independent trackers for person and chair objects
- The system SHALL avoid cross-class matching between person and chair tracks
- The system SHALL preserve track continuity across short missing-frame gaps

### FR4: Metric Depth
- The system SHALL compute or refresh a depth map on an interval
- The system SHALL reuse the latest valid depth map when a refresh frame fails
- The system SHALL clamp invalid or out-of-range depth values before conversion

### FR5: 3D Feature Generation
- The system SHALL convert detected bounding boxes into 3D spatial object grids
- The system SHALL calculate object-level 3D statistics for each instance
- The system SHALL produce a person–chair pair feature vector for regression

### FR6: Distance Prediction
- The system SHALL predict the distance between a matched person and chair in meters
- The system SHALL use the trained release model artifact when available
- The system SHALL apply temporal smoothing to reduce jitter in final estimates

### FR7: Visualization and Monitoring
- The system SHALL overlay detections, tracked IDs, and distance readouts on live frames
- The system SHALL stream the live output as MJPEG over HTTP
- The system SHALL expose runtime status and telemetry through JSON endpoints

---

## 8. Non-Functional Requirements

### NFR1: Performance
- Real-time webcam processing is required for interactive use
- The pipeline must avoid blocking the video stream while depth or ML inference runs
- Depth refresh should be throttled rather than run on every frame without need

### NFR2: Reliability
- Missing depth or feature frames must not remove detections from the current scene
- Inference must continue gracefully if a single frame fails validation
- The system must log or surface explicit errors when models are missing or invalid

### NFR3: Usability
- Local live viewing should be available without complex manual setup
- Model files and config values should be explicit and easy to change
- Visual output should be understandable during debugging and demos

### NFR4: Maintainability
- Model logic is modularized across detection, depth, grid extraction, and regression stages
- Feature contract order is fixed across `PairFeatureExtractor.feature_names()` and the exported model
- Code structure supports iterative improvement without rewriting the whole pipeline

### NFR5: Scalability
- The current architecture supports multiple objects per frame
- The system can be extended to additional object classes or new feature sets
- The design is suitable for dataset collection and regression retraining loops

---

## 9. Constraints and Assumptions

### Constraints
1. The active implementation is built around a local webcam pipeline, not a distributed cloud service
2. The product currently targets person and chair objects only
3. The pipeline depends on the presence of the YOLO model weights and the trained distance model
4. The system is primarily designed for local experimentation and deployment in a research or lab environment
5. Python 3.11 is the effective runtime in this repository

### Assumptions
1. The camera is calibrated well enough for the current intrinsics and depth conversion pipeline
2. The object detector is adequate for common indoor person/chair scenes
3. Depth estimation is used as a supporting signal, not as the sole source of truth
4. The trained regression model is periodically retrained with measured ground-truth distances
5. Users are comfortable with a local Python-based ML workflow and Flask web debugging interface

---

## 10. Current Scope vs. Out of Scope

### In Scope
- Live person and chair detection
- Person–chair distance estimation in metric units
- Independent tracking by object class
- Depth + 3D feature pipeline
- MJPEG visualization and JSON status endpoints
- Model retraining and feature refinement workflows

### Out of Scope
- Full-body SMPL mesh reconstruction from HMR2.0
- Precision motion-capture production pipeline
- Multi-camera tracking across rooms or environments
- General-purpose 3D scene reconstruction
- Commercial API or cloud deployment at this stage
- Full 4D human reconstruction workflow as the primary product goal

---

## 11. Roadmap

### Phase 1: Stability and Validation (Current)
- [x] YOLO11n person/chair detection
- [x] Independent class-specific tracking
- [x] YOLO26n depth integration
- [x] 3D object grid and feature extraction
- [x] Ridge model inference path
- [x] Flask MJPEG + API output

### Phase 2: Reliability and Calibration (Next)
- [ ] Improve depth filtering and invalid-frame handling
- [ ] Standardize calibration and metadata for camera intrinsics
- [ ] Add robust evaluation for distance accuracy over real dataset samples
- [ ] Reduce runtime jitter via stronger temporal smoothing and filtering

### Phase 3: Model Quality and Dataset Maturity
- [ ] Expand dataset coverage for more indoor layouts and edge cases
- [ ] Retrain the regression model on more validated ground-truth distances
- [ ] Test multi-object performance in crowded scenes
- [ ] Add metrics and reporting for distance error and failure modes

### Phase 4: Future Extensions
- [ ] Add additional object classes beyond chair/person if needed
- [ ] Explore more advanced regression or deep-learning distance models
- [ ] Improve UI and telemetry for monitoring during live sessions
- [ ] Consider integration with other human-scene understanding tasks

---

## 12. Success Metrics

The project is considered successful when:

1. Person and chair detections remain stable in live video for the intended use cases
2. Distance prediction stays within acceptable error bounds for the target indoor environment
3. Multiple objects can be tracked in the same frame without class confusion
4. Output remains usable in a live browser interface without crashing or stalling the stream
5. Feature and model pipelines are reproducible enough to support retraining and validation

---

## 13. Testing and Quality Assurance

### Current Validation Strategy
- Live camera testing with real frame streams
- Manual validation of object IDs and class filtering
- Feature-vector contract checks between extractor and model
- Error handling for invalid depth or incomplete frame data
- End-to-end inspection of webcam overlays and API responses

### Quality Measures
- Detection stability by class
- Temporal consistency of tracked object IDs
- Camera-frame continuity during live use
- Correctness of 3D grid and feature extraction outputs
- Regression model robustness on measured distance samples

---

## 14. Glossary

- **YOLO11n:** Object detector used for person and chair recognition
- **YOLO26n-depth:** Metric depth estimation model
- **Object3DGrid:** Spatial 3D representation of an object region
- **PairFeatureExtractor:** Creates a person–chair spatial feature vector
- **Ridge model:** Lightweight regression model used for final distance estimation
- **MJPEG:** Live streaming format used for webcam visualization

---

**Document Owner:** Development Team  
**Last Updated:** 2026-09-15  
**Next Review Date:** 2026-12-15
