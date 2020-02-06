from pathlib import Path

import nibabel


def read_image(nifti_path: Path):
    nifti_image = nibabel.load(str(nifti_path))
    image = nifti_image.get_fdata()
    return image
