from collections import deque
from typing import Any, Dict, List


class TemporalFeatureBuffer:
    """
    Stores a rolling sequence of human feature frames.

    The HAR engine will later use this sequence to recognize
    activities from motion over time instead of a single frame.
    """

    def __init__(self, max_length: int = 30):
        if max_length < 2:
            raise ValueError(
                "max_length must be at least 2"
            )

        self.max_length = max_length
        self.buffer = deque(
            maxlen=max_length
        )

    def add(
        self,
        feature_frame: Dict[str, Any],
    ) -> None:
        """
        Add one feature frame to the rolling buffer.
        """

        if not isinstance(
            feature_frame,
            dict,
        ):
            raise TypeError(
                "feature_frame must be a dictionary"
            )

        self.buffer.append(
            feature_frame
        )

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return all buffered frames in chronological order.
        """

        return list(self.buffer)

    def latest(self):
        """
        Return the newest frame.
        """

        if not self.buffer:
            return None

        return self.buffer[-1]

    def oldest(self):
        """
        Return the oldest frame.
        """

        if not self.buffer:
            return None

        return self.buffer[0]

    def size(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """
        True once enough frames are available for a
        temporal HAR decision.
        """

        return len(self.buffer) >= self.max_length

    def clear(self) -> None:
        self.buffer.clear()
