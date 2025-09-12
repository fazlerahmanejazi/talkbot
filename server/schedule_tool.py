from __future__ import annotations
from typing import List, Tuple
import re
from .calendar_mock import list_slots as _list, book_slot_by_id as _book, Slot

def list_slots(limit: int = 10) -> List[Slot]:
    return _list(limit=limit)

# ultra-simple “book by day” helper: if there’s exactly one slot that day, book it.
def book_slot(request_text: str) -> Tuple[bool, str]:
    t = (request_text or "").lower()
    day = next((d for d in ["mon","tue","wed","thu","fri","sat","sun"] if d in t), None)
    # if day found and only one slot that day, book it
    if day:
        cap = {"mon":"Mon","tue":"Tue","wed":"Wed","thu":"Thu","fri":"Fri","sat":"Sat","sun":"Sun"}[day]
        day_slots = _list(limit=50, day=cap)
        if len(day_slots) == 1:
            return _book(day_slots[0].id)
    # fall back: match a start hour token like "9", "9am", "11:00"
    m = re.search(r"\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*(am|pm)?\b", t)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2) or 0); ap = (m.group(3) or "").lower()
        for s in _list(limit=50):
            raw = s.raw.lower().replace("–", "-")
            # crude start time check
            starts = [f"{hh}:{mm:02d}", f"{hh}", f"{hh}:00"]
            if any(x in raw for x in starts) and (not ap or ap in raw):
                return _book(s.id)
    return False, "I couldn’t match that. Say a day and time, like 'Friday 9am'."
