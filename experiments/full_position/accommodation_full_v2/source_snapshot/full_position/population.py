"""Immutable evaluation schedules, constructed before candidate predictions."""
from dataclasses import dataclass
from collections import Counter

FIELDS = ("fold", "capture", "fixation", "row", "source_frame_index")


def frame_id(frame):
    values = []
    for name in FIELDS:
        value = frame[name]
        if name in ("fold", "capture"):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid scheduled {name}")
        else:
            if isinstance(value, bool) or float(value) != int(value) or int(value) < 0:
                raise ValueError(f"Invalid scheduled {name}")
            value = int(value)
        values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class PopulationManifest:
    frame_ids: tuple

    def __post_init__(self):
        if not isinstance(self.frame_ids, tuple) or any(not isinstance(k, tuple) for k in self.frame_ids):
            raise ValueError("Population manifest must contain immutable tuples")
        if len(set(self.frame_ids)) != len(self.frame_ids) or not self.frame_ids:
            raise ValueError("Scheduled population must be nonempty and unique")
        for key in self.frame_ids:
            if len(key) != len(FIELDS) or frame_id(dict(zip(FIELDS, key))) != key:
                raise ValueError("Invalid scheduled frame identity")

    @property
    def exposures(self):
        return tuple(sorted({k[:3] for k in self.frame_ids}))

    def validate(self, frames):
        keys = [frame_id(f) for f in frames]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate scheduled frame identity")
        missing, extra = set(self.frame_ids)-set(keys), set(keys)-set(self.frame_ids)
        if missing or extra:
            raise ValueError(f"Scheduled population mismatch: {len(missing)} missing, {len(extra)} unexpected frames")
        for frame in frames:
            slots = frame.get("slots", [])
            if len(slots) != 3 or Counter(s.get("held_point") for s in slots) != Counter((0, 1, 2)):
                raise ValueError("Scheduled frame requires exactly three held-P4 slots")
            for slot in slots:
                if isinstance(slot["held_point"], bool) or any(slot.get(k) != frame[k] for k in FIELDS[:-1]):
                    raise ValueError("Scheduled slot identity disagrees with its frame")
                if slot.get("source_frame_index", frame["source_frame_index"]) != frame["source_frame_index"]:
                    raise ValueError("Scheduled slot source-frame identity mismatch")
        return self


def manifest(schedule):
    """schedule must come from a fixed sampler, never surviving predictions."""
    return PopulationManifest(tuple(sorted(frame_id(f) for f in schedule)))
