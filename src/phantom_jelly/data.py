"""Loading, cleaning and leakage-safe feature selection."""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def ensure_extracted() -> None:
    """Unzip the raw archive into ``data/`` if the CSVs are not there yet."""
    if cfg.DATA_DIR.exists() and any(cfg.DATA_DIR.glob("*.csv")):
        return
    if not cfg.RAW_ZIP.exists():
        raise FileNotFoundError(
            f"Neither extracted CSVs at {cfg.DATA_DIR} nor archive at {cfg.RAW_ZIP}."
        )
    with zipfile.ZipFile(cfg.RAW_ZIP) as zf:
        zf.extractall(cfg.DATA_DIR.parent)


def _parse_tuple(series: pd.Series) -> pd.DataFrame:
    """Expand a column of ``"(a, b, c)"`` strings into one float column per element."""
    parsed = series.dropna().map(ast.literal_eval)
    if parsed.empty:
        return pd.DataFrame(index=series.index)
    width = len(parsed.iloc[0])
    out = pd.DataFrame(index=series.index, columns=range(width), dtype=float)
    out.loc[parsed.index, :] = np.asarray(parsed.tolist(), dtype=float)
    out.columns = [f"{series.name}__{i}" for i in range(width)]
    return out


def load_raw(variant: str = "reduced") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the train/test CSVs of a dataset variant with no processing applied."""
    ensure_extracted()
    try:
        train_name, test_name = cfg.DATASETS[variant]
    except KeyError as exc:
        raise ValueError(f"variant must be one of {sorted(cfg.DATASETS)}") from exc
    train = pd.read_csv(cfg.DATA_DIR / train_name)
    test = pd.read_csv(cfg.DATA_DIR / test_name)
    return train, test


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and expand the structured PyRadiomics diagnostics columns.

    The bounding-box column is the only structured column worth keeping: its last three
    elements are the ROI extent in voxels. Its first three (and the centre-of-mass
    columns) encode absolute position inside the phantom, which lets a model memorise the
    insert layout instead of learning imaging physics, so they are expanded but tagged for
    exclusion by :func:`select_features`.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    for col in ("CT_Scan_Time", "NM_Scan_Time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M", errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    structured = [
        c
        for c in df.columns
        if c.startswith("diagnostics_Mask-original_BoundingBox")
        or c.startswith("diagnostics_Mask-original_CenterOfMass")
        or c.startswith("diagnostics_Image-original_Spacing")
        or c.startswith("diagnostics_Mask-original_Spacing")
        or c.startswith("diagnostics_Image-original_Size")
        or c.startswith("diagnostics_Mask-original_Size")
    ]
    expanded = [_parse_tuple(df[c]) for c in structured]
    if expanded:
        df = pd.concat([df.drop(columns=structured), *expanded], axis=1)

    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add NM-only features motivated by SPECT physics and phantom geometry.

    All of these are computable at inference time from the NM acquisition alone, so none
    of them touch the CT segmentation.
    """
    df = df.copy()
    counts = df.get(cfg.NM_COUNTS_COLUMN)
    volume = df.get(cfg.NM_VOLUME_COLUMN)

    if counts is not None:
        # Total counts in the ROI: the extensive counterpart of mean intensity.
        if volume is not None:
            df["feat_total_counts"] = counts * volume
        df["feat_log_counts"] = np.log1p(counts.clip(lower=0))

    if volume is not None:
        df["feat_log_volume_nm"] = np.log1p(volume.clip(lower=0))
        # Equivalent-sphere radius: linearises the volume/surface relationship.
        df["feat_equiv_radius_nm"] = np.cbrt(volume.clip(lower=0) * 3.0 / (4.0 * np.pi))

    surface = df.get("original_shape_SurfaceArea_NM_40")
    mesh = df.get("original_shape_MeshVolume_NM_40")
    if surface is not None and mesh is not None:
        # Partial-volume effect scales with the surface-to-volume ratio, which is the
        # dominant driver of the NM-vs-CT volume bias.
        df["feat_pve_proxy"] = surface / mesh.replace(0, np.nan)

    if mesh is not None and volume is not None:
        df["feat_mesh_voxel_ratio"] = mesh / volume.replace(0, np.nan)

    return df


# --------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------
POSITIONAL_MARKERS = ("CenterOfMass", "BoundingBox__0", "BoundingBox__1", "BoundingBox__2")


def _is_ct_derived(col: str) -> bool:
    return col.endswith("_CT") or col in cfg.OVERLAP_COLUMNS


def _is_diagnostic_string(col: str) -> bool:
    return any(col.startswith(p) for p in cfg.DIAGNOSTIC_PREFIXES)


def _is_positional(col: str) -> bool:
    return any(m in col for m in POSITIONAL_MARKERS)


@dataclass
class FeatureSelection:
    """The outcome of feature selection, with an audit trail of what was removed."""

    features: list[str]
    dropped: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"kept {len(self.features)} features"]
        for reason, cols in self.dropped.items():
            lines.append(f"  dropped {len(cols):3d}  ({reason})")
        return "\n".join(lines)


# The two columns that carry the physical signal for the two targets. They are exempt
# from the redundancy filter so that the physics baselines always have their input, and so
# that a near-duplicate radiomics feature never displaces the interpretable one.
PROTECTED_FEATURES = (cfg.NM_COUNTS_COLUMN, cfg.NM_VOLUME_COLUMN)


def select_features(
    train: pd.DataFrame,
    *,
    allow_ct_radiomics: bool = False,
    allow_nm80: bool = True,
    allow_positional: bool = False,
    corr_threshold: float | None = 0.995,
    protected: tuple[str, ...] = PROTECTED_FEATURES,
) -> FeatureSelection:
    """Choose the modelling features from a cleaned training frame.

    Parameters
    ----------
    allow_ct_radiomics:
        Off by default. CT-derived columns are a direct leak of ``Volume_[ml]_CT`` and are
        excluded from the reduced dataset by design; only enable this behind a validated
        surrogate model (see :mod:`phantom_jelly.surrogate`).
    corr_threshold:
        Drop one of every pair of features whose absolute Pearson correlation exceeds this
        value. Radiomics packs contain many exact algebraic duplicates.
    protected:
        Columns that survive the redundancy filter regardless of their correlations.
    """
    dropped: dict[str, list[str]] = {}

    def record(reason: str, cols) -> None:
        cols = sorted(set(cols))
        if cols:
            dropped[reason] = cols

    candidates = list(train.columns)

    targets = [c for c in cfg.TARGETS if c in candidates]
    record("target", targets)
    candidates = [c for c in candidates if c not in targets]

    ids = [c for c in candidates if c in cfg.ID_COLUMNS]
    record("segmentation label id", ids)
    candidates = [c for c in candidates if c not in ids]

    times = [c for c in candidates if c in cfg.TIME_COLUMNS]
    record("timestamp / acquisition key", times)
    candidates = [c for c in candidates if c not in times]

    diagnostics = [c for c in candidates if _is_diagnostic_string(c)]
    record("pyradiomics bookkeeping", diagnostics)
    candidates = [c for c in candidates if c not in diagnostics]

    if not allow_ct_radiomics:
        ct = [c for c in candidates if _is_ct_derived(c)]
        record("CT-derived (leaks target 1)", ct)
        candidates = [c for c in candidates if c not in ct]

    if not allow_nm80:
        nm80 = [c for c in candidates if c.endswith("_NM_80")]
        record("80% NM segmentation", nm80)
        candidates = [c for c in candidates if c not in nm80]

    if not allow_positional:
        positional = [c for c in candidates if _is_positional(c)]
        record("absolute position in phantom", positional)
        candidates = [c for c in candidates if c not in positional]

    numeric = train[candidates].select_dtypes(include=[np.number])
    record("non-numeric", set(candidates) - set(numeric.columns))
    candidates = list(numeric.columns)

    constant = [c for c in candidates if train[c].nunique(dropna=False) <= 1]
    record("constant", constant)
    candidates = [c for c in candidates if c not in constant]

    if corr_threshold is not None and len(candidates) > 1:
        # Order protected columns first so the filter keeps them and drops their duplicates.
        ordered = [c for c in protected if c in candidates]
        ordered += [c for c in candidates if c not in ordered]
        corr = train[ordered].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        redundant = [
            c for c in upper.columns if c not in protected and (upper[c] > corr_threshold).any()
        ]
        record(f"|r| > {corr_threshold} with an earlier feature", redundant)
        candidates = [c for c in ordered if c not in redundant]

    return FeatureSelection(features=candidates, dropped=dropped)


# --------------------------------------------------------------------------------------
# Convenience bundle
# --------------------------------------------------------------------------------------
@dataclass
class Dataset:
    """A ready-to-model train/test pair plus the grouping keys used for validation.

    Two grouping levels are carried around. ``groups_train`` is the acquisition (NM image
    hash) and is the minimum needed to avoid leaking a per-scan constant target.
    ``sessions_train`` is the acquisition date; because the supplied test file is an
    entirely unseen session, session-level CV is the estimate that actually matches the
    deployment scenario.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    sessions_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    groups_test: pd.Series
    sessions_test: pd.Series
    selection: FeatureSelection
    target: str

    @property
    def n_groups(self) -> int:
        return int(self.groups_train.nunique())

    @property
    def n_sessions(self) -> int:
        return int(self.sessions_train.nunique())


def build_dataset(
    target: str,
    variant: str = "reduced",
    *,
    allow_ct_radiomics: bool = False,
    allow_nm80: bool = True,
    physics_features: bool = True,
    group_key: str = cfg.SCAN_KEY,
    corr_threshold: float | None = 0.995,
) -> Dataset:
    """Load, clean and split a dataset variant for a single target."""
    train_raw, test_raw = load_raw(variant)
    groups_train = train_raw[group_key].copy()
    groups_test = test_raw[group_key].copy()
    sessions_train = train_raw[cfg.SESSION_KEY].astype(str).copy()
    sessions_test = test_raw[cfg.SESSION_KEY].astype(str).copy()

    train, test = clean(train_raw), clean(test_raw)
    if physics_features:
        train, test = add_physics_features(train), add_physics_features(test)

    selection = select_features(
        train,
        allow_ct_radiomics=allow_ct_radiomics,
        allow_nm80=allow_nm80,
        corr_threshold=corr_threshold,
    )
    features = [c for c in selection.features if c in test.columns]

    return Dataset(
        X_train=train[features],
        y_train=train[target],
        groups_train=groups_train.reset_index(drop=True),
        sessions_train=sessions_train.reset_index(drop=True),
        X_test=test[features],
        y_test=test[target],
        groups_test=groups_test.reset_index(drop=True),
        sessions_test=sessions_test.reset_index(drop=True),
        selection=FeatureSelection(features, selection.dropped),
        target=target,
    )
