import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian, generate_uid

from knee_mri.data.dicom import (
    load_study_volumes,
    plane_from_orientation,
    scan_study,
    select_primary_series,
)

# Row/col direction cosines per plane (slice normal → dominant axis).
_ORIENTATIONS = {
    "axial": [1, 0, 0, 0, 1, 0],       # normal ~ z
    "coronal": [1, 0, 0, 0, 0, -1],    # normal ~ y
    "sagittal": [0, 1, 0, 0, 0, -1],   # normal ~ x
}


def _write_series(series_dir, plane, description, n_slices, size=16, reverse=False):
    series_dir.mkdir(parents=True)
    iop = _ORIENTATIONS[plane]
    normal = np.cross(iop[:3], iop[3:])
    order = range(n_slices - 1, -1, -1) if reverse else range(n_slices)
    for out_idx, z in enumerate(order):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesDescription = description
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.ImagePositionPatient = [float(v) for v in normal * float(z) * 3.0]
        ds.InstanceNumber = z + 1
        ds.Rows = ds.Columns = size
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        # Encode slice index into the pixels so sort order is checkable.
        ds.PixelData = np.full((size, size), z, dtype=np.uint16).tobytes()
        ds.save_as(series_dir / f"{out_idx:03d}.dcm", enforce_file_format=True)


@pytest.fixture()
def dicom_study(tmp_path):
    study = tmp_path / "study-0001"
    # Two axial series: T1 (more slices) vs fluid-sensitive PD FS (fewer) —
    # selection must prefer the fluid-sensitive one.
    _write_series(study / "s1", "axial", "AX T1", n_slices=8)
    _write_series(study / "s2", "axial", "AX PD FS", n_slices=5, reverse=True)
    _write_series(study / "s3", "coronal", "COR STIR", n_slices=6)
    _write_series(study / "s4", "sagittal", "SAG T2 FS", n_slices=7)
    return study


def test_plane_from_orientation():
    for plane, iop in _ORIENTATIONS.items():
        assert plane_from_orientation([float(v) for v in iop]) == plane
    assert plane_from_orientation(None) is None


def test_scan_and_primary_selection(dicom_study):
    infos = scan_study(dicom_study)
    assert len(infos) == 4
    primary = select_primary_series(infos)
    assert set(primary) == {"axial", "coronal", "sagittal"}
    # Fluid-sensitive series wins over the longer T1.
    assert primary["axial"].description == "AX PD FS"


def test_volume_loading_sorts_slices(dicom_study):
    volumes = load_study_volumes(dicom_study)
    assert set(volumes) == {"axial", "coronal", "sagittal"}
    # Files were written in reverse order; sorting by ImagePositionPatient
    # must restore ascending slice order (pixel value == slice index).
    axial = volumes["axial"]
    assert axial.shape == (5, 16, 16)
    assert [int(axial[i, 0, 0]) for i in range(5)] == [0, 1, 2, 3, 4]
