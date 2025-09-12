from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Slot:
    id: str
    raw: str  # e.g. "Fri Sep 19, 9:00–9:30 AM PT"
    day: str  # "Mon".."Sun"

# seed deterministic slots
_SLOTS: List[Slot] = [
    Slot("tuesday-1400", "Tuesday Sep 16, 2:00–2:30 PM PT", "Tuesday"),
    Slot("wednesday-1100", "Wednesday Sep 17, 11:00–11:30 AM PT", "Wednesday"),
    Slot("thursday-1630", "Thursday Sep 18, 4:30–5:00 PM PT", "Thursday"),
    Slot("friday-0900", "Friday Sep 19, 9:00–9:30 AM PT", "Friday"),
]

_BOOKED: set[str] = set()

def list_slots(limit: int = 10, day: Optional[str] = None) -> List[Slot]:
    xs = [s for s in _SLOTS if s.id not in _BOOKED and (day is None or s.day == day)]
    return xs[: max(1, min(limit, 50))]

def book_slot_by_id(slot_id: str) -> tuple[bool, str]:
    for s in _SLOTS:
        if s.id == slot_id and s.id not in _BOOKED:
            _BOOKED.add(slot_id)
            return True, f"Booked: {s.raw}"
    return False, "That slot isn’t available."
