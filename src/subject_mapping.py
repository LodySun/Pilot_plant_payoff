"""Published subject correspondence for the two Sherlock releases.

Born et al. (2023), Data in Brief 50:109490, Table 1, explicitly maps
Princeton DataSpace subjects (used by the 17-subject Chen ROI package) to
OpenNeuro subjects (used by the 16-subject fMRIPrep release). Princeton
subject 05 is the sole omission from OpenNeuro.

This mapping permits paired cross-release checks. It does not justify pooling
duplicate measurements from the two releases as independent observations.
"""
from __future__ import annotations

MAPPING_DOI = "10.1016/j.dib.2023.109490"

OPENNEURO_TO_PRINCETON: dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
}

PRINCETON_TO_OPENNEURO: dict[int, int | None] = {
    subject: (subject if subject <= 4 else None if subject == 5 else subject - 1)
    for subject in range(1, 18)
}

# OpenNeuro retains five initial recall volumes that are absent from the
# Princeton/DataSpace processed time base. Born et al.'s timestamp files give
# both TR columns; their difference is exactly +5 for every word and subject.
OPENNEURO_RECALL_OFFSET_TRS = 5


def openneuro_id(princeton_id: int) -> int | None:
    """Return the published OpenNeuro ID, or None for omitted subject 05."""
    return PRINCETON_TO_OPENNEURO[int(princeton_id)]


def princeton_id(openneuro_id: int) -> int:
    """Return the published Princeton/Chen ID for an OpenNeuro subject."""
    return OPENNEURO_TO_PRINCETON[int(openneuro_id)]
