# Find all projects (*.bsprj.json) in a folder
# and extract all series (images and masks) (.nii.gz) with non-empty masks

import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_PATH = Path(__file__).parents[4]
DATA_PATH = PROJECT_PATH / 'data'
MS_UNFILTERED_DATA_PATH = DATA_PATH / 'ms-lesions-unfiltered'
MS_DATA_PATH = DATA_PATH / 'ms-lesions'
UNZIP_TEMP_PATH = DATA_PATH / 'temp'

BS_PROJECT_SUFFIX = '.bsprj.zip'
BS_PROJECT_CONFIG_SUFFIX = '.bsprj.json'


def projects_in_dir(dir_path: Path, recursively: bool = True):
    for child_path in dir_path.iterdir():
        if child_path.is_dir() and recursively:
            yield from projects_in_dir(child_path, recursively)
        elif child_path.name.endswith(BS_PROJECT_SUFFIX):
            yield child_path


def extract_project_series(project_path: Path, extracted_data_path: Path, unzip_temp_path: Path):
    unzipped_project_path = unzip_bsprj(project_path, unzip_temp_path)
    copy_series(unzipped_project_path, extracted_data_path)


def unzip_bsprj(project_path: Path, unzip_dir_path: Path):
    with zipfile.ZipFile(project_path, 'r') as zip_ref:
        project_name = project_path.name[:-len(BS_PROJECT_SUFFIX)]  # without extension
        unzip_path = unzip_dir_path / project_name
        zip_ref.extractall(path=unzip_path)
        return unzip_path


def copy_series(unzipped_project_path: Path, dst_dir_path: Path):
    project_config_files = list(unzipped_project_path.glob(f'*{BS_PROJECT_CONFIG_SUFFIX}'))
    assert len(project_config_files) == 1, 'Project has to contain one and only one config file (*.bsprj.json)'
    project_config_path = project_config_files[0]
    with open(str(project_config_path)) as project_config_ref:
        project_config = json.load(project_config_ref)
        all_series = project_config['series']

        for syncer in project_config['syncers'].values():
            series_id = syncer['seriesId']
            mask_name = syncer['maskPath']
            name = syncer['name']

            mask_path = unzipped_project_path / mask_name
            mask_nifti = nib.load(mask_path)
            mask = np.asanyarray(mask_nifti.dataobj)

            series_name = f'{unzipped_project_path.name}___{create_valid_filename(name)}.nii.gz'

            if mask.max() > 0:
                # Copy series mask and image
                shutil.copyfile(mask_path, dst_dir_path / 'masks' / series_name)

                image_name = all_series[str(series_id)]['filePath']
                imape_path = unzipped_project_path / image_name

                shutil.copyfile(imape_path, dst_dir_path / 'series' / series_name)


def create_valid_filename(string: str):
    return ''.join(c for c in string if c not in r'\/:*?<>|')


def main():
    print(f'START EXTRACTOR   Project_path: {PROJECT_PATH}')
    for project_path in projects_in_dir(MS_UNFILTERED_DATA_PATH):
        print(f'Project {project_path}')
        extract_project_series(project_path, MS_DATA_PATH, UNZIP_TEMP_PATH)


if __name__ == '__main__':
    main()
