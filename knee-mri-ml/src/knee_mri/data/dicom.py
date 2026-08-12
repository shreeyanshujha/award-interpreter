"""DICOM ingestion for RSNA-style knee MRI studies.

Real studies arrive as ``<study>/<series>/*.dcm`` with 3–14 series spanning
the three anatomical planes. This module:

1. groups files into series and reads lightweight metadata,
2. infers each series' plane from ``ImageOrientationPatient``,
3. picks one *primary* series per plane, preferring fluid-sensitive
   fat-suppressed sequences (where pathology is most conspicuous),
4. sorts slices along the slice normal via ``ImagePositionPatient``,
5. returns ``{plane: (num_slices, H, W) float32 array}``.

pydicom is imported lazily so the rest of the package (synthetic/.npy paths)
works without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PLANES = ("axial", "coronal", "sagittal")

# SeriesDescription fragments that indicate fluid-sensitive / fat-suppressed
# sequences (preferred for reading most knee pathology).
_FLUID_SENSITIVE = ("stir", "spair", "fs", "fatsat", "fat sat", "fat-sat", "t2", "pd")


def _require_pydicom():
    try:
        import pydicom  # noqa: F401

        return pydicom
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pydicom is required for DICOM ingestion: pip install pydicom"
        ) from exc


def plane_from_orientation(iop: Optional[List[float]]) -> Optional[str]:
    """Classify a series' plane from the 6-value ImageOrientationPatient.

    The slice normal is the cross product of the row and column direction
    cosines; its dominant axis determines the plane (x → sagittal,
    y → coronal, z → axial).
    """
    if iop is None or len(iop) != 6:
        return None
    row, col = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    normal = np.cross(row, col)
    axis = int(np.argmax(np.abs(normal)))
    return ("sagittal", "coronal", "axial")[axis]


@dataclass
class SeriesInfo:
    """Lightweight metadata for one DICOM series."""

    series_dir: Path
    plane: Optional[str] = None
    description: str = ""
    num_files: int = 0
    files: List[Path] = field(default_factory=list)

    @property
    def fluid_sensitive(self) -> bool:
        desc = self.description.lower()
        return any(tag in desc for tag in _FLUID_SENSITIVE)


def scan_study(study_dir: str | Path) -> List[SeriesInfo]:
    """Read per-series metadata for every series directory in a study."""
    pydicom = _require_pydicom()
    study_dir = Path(study_dir)
    series_dirs = sorted(d for d in study_dir.iterdir() if d.is_dir())
    if not series_dirs:
        # Flat layout: treat the study directory itself as a single series.
        series_dirs = [study_dir]

    infos: List[SeriesInfo] = []
    for sdir in series_dirs:
        files = sorted(sdir.glob("*.dcm")) or sorted(
            f for f in sdir.iterdir() if f.is_file()
        )
        if not files:
            continue
        # One header read is enough for orientation + description.
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        iop = getattr(ds, "ImageOrientationPatient", None)
        infos.append(
            SeriesInfo(
                series_dir=sdir,
                plane=plane_from_orientation(list(iop) if iop is not None else None),
                description=str(getattr(ds, "SeriesDescription", "") or ""),
                num_files=len(files),
                files=files,
            )
        )
    return infos


def select_primary_series(infos: List[SeriesInfo]) -> Dict[str, SeriesInfo]:
    """Pick one series per plane: fluid-sensitive first, then most slices."""
    primary: Dict[str, SeriesInfo] = {}
    for plane in PLANES:
        candidates = [s for s in infos if s.plane == plane]
        if not candidates:
            continue
        candidates.sort(key=lambda s: (s.fluid_sensitive, s.num_files), reverse=True)
        primary[plane] = candidates[0]
    return primary


def load_series_volume(info: SeriesInfo) -> np.ndarray:
    """Load a series' pixel data sorted along the slice normal."""
    pydicom = _require_pydicom()

    slices = []
    for f in info.files:
        ds = pydicom.dcmread(f)
        pos = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if pos is not None and iop is not None and len(iop) == 6:
            row, col = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
            normal = np.cross(row, col)
            order_key = float(np.dot(np.asarray(pos, float), normal))
        else:
            # Fall back to InstanceNumber, then filename order.
            order_key = float(getattr(ds, "InstanceNumber", 0) or 0)
        slices.append((order_key, ds.pixel_array.astype(np.float32)))

    slices.sort(key=lambda t: t[0])
    return np.stack([px for _, px in slices])


def load_study_volumes(study_dir: str | Path) -> Dict[str, np.ndarray]:
    """Full pipeline: scan → select primary per plane → load sorted volumes."""
    infos = scan_study(study_dir)
    primary = select_primary_series(infos)
    if not primary:
        raise ValueError(f"no readable DICOM series found under {study_dir}")
    return {plane: load_series_volume(info) for plane, info in primary.items()}
