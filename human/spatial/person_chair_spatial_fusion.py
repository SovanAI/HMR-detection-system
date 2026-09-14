"""
BAS HMR - Person ↔ Chair Spatial Fusion

Combines:

    Person detection
          ↓
        HMR2
          ↓
    Person X,Y,Z

    Chair detection
          ↓
    Depth Anything
          ↓
    Depth calibration
          ↓
      Chair Z

Then calculates:

    Person ↔ Chair spatial distance

IMPORTANT:
    Distance is currently in the calibrated
    HMR-compatible coordinate system.

    It is NOT guaranteed to be metres.
"""


import json
import math
import os
from dataclasses import dataclass
from typing import Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Point3D:

    x: float
    y: float
    z: float


@dataclass
class PersonSpatial:

    person_id: int

    position: Point3D

    bbox: dict

    confidence: float


@dataclass
class ChairSpatial:

    chair_id: int

    position: Point3D

    bbox: dict

    confidence: float

    relative_depth: float


@dataclass
class SpatialRelationship:

    person_id: int

    chair_id: int

    dx: float

    dy: float

    dz: float

    distance: float

    relationship: str


# ============================================================
# DEPTH → HMR CALIBRATION
# ============================================================

class DepthHMRCalibration:

    def __init__(
        self,
        scale: float,
        offset: float
    ):

        self.scale = float(
            scale
        )

        self.offset = float(
            offset
        )

    # --------------------------------------------------------
    # Convert relative depth to HMR Z
    # --------------------------------------------------------

    def depth_to_hmr_z(
        self,
        relative_depth: float
    ) -> float:

        return (
            self.scale
            * float(relative_depth)
            + self.offset
        )

    # --------------------------------------------------------
    # Load calibration file
    # --------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str
    ):

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Calibration file not found: {path}"
            )

        with open(
            path,
            "r"
        ) as f:

            data = json.load(f)

        scale = data.get(
            "scale"
        )

        offset = data.get(
            "offset"
        )

        if scale is None or offset is None:

            raise ValueError(
                "Calibration file does not contain "
                "scale and offset."
            )

        return cls(
            scale=scale,
            offset=offset
        )


# ============================================================
# PERSON EXTRACTOR
# ============================================================

class PersonSpatialExtractor:

    @staticmethod
    def extract(
        person: dict
    ) -> Optional[PersonSpatial]:

        translation = person.get(
            "camera_translation"
        )

        if not translation:

            return None

        person_id = int(
            person.get(
                "person_id",
                0
            )
        )

        position = Point3D(

            x=float(
                translation["x"]
            ),

            y=float(
                translation["y"]
            ),

            z=float(
                translation["z"]
            ),
        )

        bbox = person.get(
            "bbox",
            {}
        )

        confidence = float(
            person.get(
                "confidence",
                1.0
            )
        )

        return PersonSpatial(

            person_id=person_id,

            position=position,

            bbox=bbox,

            confidence=confidence,
        )


# ============================================================
# CHAIR EXTRACTOR
# ============================================================

class ChairSpatialExtractor:

    @staticmethod
    def extract(
        chair: dict,
        calibration: DepthHMRCalibration,
        frame_width: int = 640,
        frame_height: int = 480
    ) -> Optional[ChairSpatial]:

        bbox = chair.get(
            "bbox"
        )

        if not bbox:

            return None

        relative_depth = chair.get(
            "relative_depth"
        )

        if relative_depth is None:

            return None

        x1 = float(
            bbox["x1"]
        )

        y1 = float(
            bbox["y1"]
        )

        x2 = float(
            bbox["x2"]
        )

        y2 = float(
            bbox["y2"]
        )

        # ----------------------------------------------------
        # Chair image center
        # ----------------------------------------------------

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        # ----------------------------------------------------
        # Normalize image coordinates
        # ----------------------------------------------------

        normalized_x = (
            center_x
            / float(frame_width)
        )

        normalized_y = (
            center_y
            / float(frame_height)
        )

        # ----------------------------------------------------
        # Convert relative depth
        # to HMR-compatible Z
        # ----------------------------------------------------

        chair_z = calibration.depth_to_hmr_z(
            relative_depth
        )

        # ----------------------------------------------------
        # Convert normalized image position
        # into the same approximate spatial space.
        #
        # IMPORTANT:
        # This is a first-stage spatial representation.
        # It is NOT yet a metric camera reconstruction.
        # ----------------------------------------------------

        chair_x = (
            normalized_x
            - 0.5
        )

        chair_y = (
            normalized_y
            - 0.5
        )

        chair_id = int(
            chair.get(
                "chair_id",
                0
            )
        )

        confidence = float(
            chair.get(
                "confidence",
                1.0
            )
        )

        return ChairSpatial(

            chair_id=chair_id,

            position=Point3D(

                x=chair_x,

                y=chair_y,

                z=chair_z,
            ),

            bbox=bbox,

            confidence=confidence,

            relative_depth=float(
                relative_depth
            ),
        )


# ============================================================
# SPATIAL ENGINE
# ============================================================

class PersonChairSpatialEngine:

    def __init__(
        self,
        near_threshold: float = 2.0,
        medium_threshold: float = 8.0
    ):

        self.near_threshold = float(
            near_threshold
        )

        self.medium_threshold = float(
            medium_threshold
        )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    @staticmethod
    def distance(
        person: Point3D,
        chair: Point3D
    ) -> float:

        dx = (
            person.x
            - chair.x
        )

        dy = (
            person.y
            - chair.y
        )

        dz = (
            person.z
            - chair.z
        )

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

    # --------------------------------------------------------
    # Relationship
    # --------------------------------------------------------

    def classify(
        self,
        distance: float
    ) -> str:

        if distance <= self.near_threshold:

            return "near"

        if distance <= self.medium_threshold:

            return "medium"

        return "far"

    # --------------------------------------------------------
    # Compare one person against one chair
    # --------------------------------------------------------

    def compare(
        self,
        person: PersonSpatial,
        chair: ChairSpatial
    ) -> SpatialRelationship:

        dx = (
            person.position.x
            - chair.position.x
        )

        dy = (
            person.position.y
            - chair.position.y
        )

        dz = (
            person.position.z
            - chair.position.z
        )

        distance = self.distance(
            person.position,
            chair.position
        )

        relationship = self.classify(
            distance
        )

        return SpatialRelationship(

            person_id=person.person_id,

            chair_id=chair.chair_id,

            dx=dx,

            dy=dy,

            dz=dz,

            distance=distance,

            relationship=relationship,
        )


# ============================================================
# FUSION ENGINE
# ============================================================

class PersonChairFusion:

    def __init__(
        self,
        calibration: DepthHMRCalibration,
        spatial_engine: PersonChairSpatialEngine
    ):

        self.calibration = calibration

        self.spatial_engine = spatial_engine

    # --------------------------------------------------------
    # Fuse
    # --------------------------------------------------------

    def fuse(
        self,
        hmr_persons: list,
        chairs: list,
        frame_width: int = 640,
        frame_height: int = 480
    ):

        persons = []

        for person in hmr_persons:

            extracted = (
                PersonSpatialExtractor.extract(
                    person
                )
            )

            if extracted is not None:

                persons.append(
                    extracted
                )

        chair_objects = []

        for chair in chairs:

            extracted = (
                ChairSpatialExtractor.extract(
                    chair,
                    self.calibration,
                    frame_width,
                    frame_height
                )
            )

            if extracted is not None:

                chair_objects.append(
                    extracted
                )

        relationships = []

        for person in persons:

            for chair in chair_objects:

                relationship = (
                    self.spatial_engine.compare(
                        person,
                        chair
                    )
                )

                relationships.append(
                    relationship
                )

        return (
            persons,
            chair_objects,
            relationships
        )


# ============================================================
# JSON SERIALIZER
# ============================================================

def relationship_to_dict(
    relationship: SpatialRelationship
):

    return {

        "person_id": (
            relationship.person_id
        ),

        "chair_id": (
            relationship.chair_id
        ),

        "delta": {

            "x": relationship.dx,

            "y": relationship.dy,

            "z": relationship.dz,
        },

        "distance": (
            relationship.distance
        ),

        "relationship": (
            relationship.relationship
        ),

        "distance_units": (
            "HMR-compatible coordinate units"
        ),

        "metric_distance": False,
    }


# ============================================================
# SELF TEST
# ============================================================

def self_test():

    print()
    print("=" * 70)
    print(
        "PERSON ↔ CHAIR SPATIAL FUSION SELF TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Use the current calibration values.
    # --------------------------------------------------------

    calibration = DepthHMRCalibration(
        scale=-40.219,
        offset=49.882
    )

    # --------------------------------------------------------
    # Example HMR person
    # --------------------------------------------------------

    hmr_persons = [

        {

            "person_id": 0,

            "camera_translation": {

                "x": -0.086,

                "y": 0.783,

                "z": 18.446,
            },

            "bbox": {

                "x1": 100,

                "y1": 100,

                "x2": 400,

                "y2": 480,
            },

            "confidence": 0.95,
        }
    ]

    # --------------------------------------------------------
    # Example tracked chair
    # --------------------------------------------------------

    chairs = [

        {

            "chair_id": 0,

            "bbox": {

                "x1": 200,

                "y1": 250,

                "x2": 500,

                "y2": 480,
            },

            "confidence": 0.90,

            "relative_depth": 0.78,
        }
    ]

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    fusion = PersonChairFusion(

        calibration=calibration,

        spatial_engine=(
            PersonChairSpatialEngine(
                near_threshold=5.0,
                medium_threshold=12.0
            )
        )
    )

    (
        persons,
        chairs_out,
        relationships
    ) = fusion.fuse(
        hmr_persons,
        chairs
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    for person in persons:

        print()
        print(
            f"Person #{person.person_id}"
        )

        print(
            f"  X = "
            f"{person.position.x:.4f}"
        )

        print(
            f"  Y = "
            f"{person.position.y:.4f}"
        )

        print(
            f"  Z = "
            f"{person.position.z:.4f}"
        )

    for chair in chairs_out:

        print()
        print(
            f"Chair #{chair.chair_id}"
        )

        print(
            f"  X = "
            f"{chair.position.x:.4f}"
        )

        print(
            f"  Y = "
            f"{chair.position.y:.4f}"
        )

        print(
            f"  Z = "
            f"{chair.position.z:.4f}"
        )

        print(
            f"  Relative Depth = "
            f"{chair.relative_depth:.4f}"
        )

    for relationship in relationships:

        print()
        print(
            "RELATIONSHIP"
        )

        print(
            f"  Person: "
            f"#{relationship.person_id}"
        )

        print(
            f"  Chair: "
            f"#{relationship.chair_id}"
        )

        print(
            f"  ΔX = "
            f"{relationship.dx:.4f}"
        )

        print(
            f"  ΔY = "
            f"{relationship.dy:.4f}"
        )

        print(
            f"  ΔZ = "
            f"{relationship.dz:.4f}"
        )

        print(
            f"  Distance = "
            f"{relationship.distance:.4f}"
        )

        print(
            f"  Relationship = "
            f"{relationship.relationship}"
        )

        print(
            "  Units = "
            "HMR-compatible coordinate units"
        )

    print()
    print("=" * 70)
    print("SELF TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    self_test()
