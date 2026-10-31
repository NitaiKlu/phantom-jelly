"""Column taxonomy, paths and physical constants for the jelly-phantom study.

Everything that encodes *domain knowledge about the dataset* lives here so that the
modelling code stays generic. The most important part is :data:`LEAK_COLUMNS`, which
lists the columns that must never be handed to a model.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "Jelly_Phantom"
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "Jelly_Phantom.zip"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------------------
# Dataset variants
# --------------------------------------------------------------------------------------
# "reduced" (train.csv / test.csv)  -> NM-40 radiomics only, small ROIs already filtered.
# "full"    (df_train.csv / df_test.csv) -> additionally NM-80 and CT radiomics + overlaps.
DATASETS = {
    "reduced": ("train.csv", "test.csv"),
    "full": ("df_train.csv", "df_test.csv"),
}

TARGETS = ("Volume_[ml]_CT", "Activity_Density_uCiml_NM_40")

# --------------------------------------------------------------------------------------
# Grouping / identifier columns
# --------------------------------------------------------------------------------------
# One acquisition == one unique NM image hash. Activity density is (almost) a per-scan
# constant, so every honest validation split has to be grouped on this key.
SCAN_KEY = "diagnostics_Image-original_Hash_NM_40"
SESSION_KEY = "date"

# Arbitrary connected-component label ids. They correlate with position in the phantom
# and therefore with the inserts, but carry no transferable signal.
ID_COLUMNS = ("NM_40", "NM_80", "CT", "Label_CT", "Label_NM_40", "Label_NM_80")

# Timestamps identify the acquisition uniquely; combined with a per-scan constant target
# they are a pure lookup key, not a feature.
TIME_COLUMNS = ("CT_Scan_Time", "NM_Scan_Time", "date")

# --------------------------------------------------------------------------------------
# Leakage blacklist
# --------------------------------------------------------------------------------------
# Any column suffixed `_CT` is derived from the CT image or the CT segmentation.
# `Volume_[ml]_CT` *is* target #1 and every CT shape feature is a deterministic function
# of the very mask that defines it, so all of them leak target #1 outright. The suffix
# test in `data._is_ct_derived` is what enforces this.
#
# Overlap features are computed between the CT mask and the NM masks, i.e. they require
# knowledge of the CT segmentation at inference time.
OVERLAP_COLUMNS = (
    "NM40_to_NM80_overlap",
    "CT_to_NM40_overlap",
    "CT_to_NM80_overlap",
)

# PyRadiomics bookkeeping: constant strings (library versions, settings) or hashes.
DIAGNOSTIC_PREFIXES = (
    "diagnostics_Versions_",
    "diagnostics_Configuration_",
    "diagnostics_Image-original_Hash",
    "diagnostics_Mask-original_Hash",
    "diagnostics_Image-original_Dimensionality",
)

# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------
# Reconstructed NM counts scale linearly with activity concentration. The proportionality
# constant is stable across acquisition sessions (~2.9e-5 uCi/ml per count), which makes
# a one-parameter calibration a genuinely strong baseline for target #2.
NM_COUNTS_COLUMN = "Mean_Intensity_counts_NM_40"
NM_VOLUME_COLUMN = "Volume_[ml]_NM_40"
SCALE_FACTOR_COLUMN = "Scale_factor_NM_40"
SETTINGS_COLUMN = "diagnostics_Configuration_Settings_NM_40"

# PyRadiomics intensity discretisation used for every NM-40 feature in the CSVs.
RADIOMICS_BIN_WIDTH = 2000

RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Experimental sessions (manuscript Table 1)
# --------------------------------------------------------------------------------------
# Chronological labels as in the original write-up. Session 0 is the held-out
# preparation. Segment counts, scan counts and activity ranges are recomputed from
# the full CSVs; jar counts are the only field that is not stored in the tables and
# are taken from the experimental log.
SESSION_BY_ISO_DATE = {
    "2025-05-20": 1,
    "2025-06-10": 2,
    "2025-07-01": 3,
    "2025-07-08": 4,
    "2025-07-21": 5,
    "2025-09-02": 0,
}
SESSION_JARS = {1: 3, 2: 3, 3: 3, 4: 4, 5: 4, 0: 4}
EXPECTED_VALIDATED_SEGMENTS = 2302
EXPECTED_SCANS = 75
EXPECTED_JARS = 21
EXPECTED_SESSIONS = 6


def ct_radiomics_columns(columns) -> list[str]:
    """Return every column that is derived from the CT image or the CT mask."""
    return [c for c in columns if c.endswith("_CT") or c in OVERLAP_COLUMNS]


def nm80_columns(columns) -> list[str]:
    """Return every column derived from the 80% NM segmentation."""
    return [c for c in columns if c.endswith("_NM_80")]
