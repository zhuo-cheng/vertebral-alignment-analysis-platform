import os
import csv
import cv2
import numpy as np
import SimpleITK as sitk
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import LandmarkActor


def load_dict_csv(file_name, value_type=str, squeeze=True):
    """
    Loads a .csv file as a dict, where the first column indicate the key string
    and the following columns are the corresponding value or list of values.
    :param file_name: The file name to load.
    :param value_type: Each value will be converted to this type.
    :param squeeze: If true, reduce single entry list to a value.
    :return: A dictionary of every entry of the .csv file.
    """
    d = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_id = row[0]
            value = list(map(value_type, row[1:]))
            if squeeze and len(value) == 1:
                value = value[0]
            d[data_id] = value
    return d



pred = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\vertebrae_segmentation_bayesian_reorient\Patient001\Patient001_pred_seg.nii.gz'
uncertainty = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\vertebrae_segmentation_bayesian_reorient\Patient001\Patient001_pred_uncert.nii.gz'

landmark_path = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\vertebrae_localization\landmarks.csv'
valid_landmark_path = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\vertebrae_localization\valid_landmarks.csv'


img = sitk.ReadImage(uncertainty)
spacing = img.GetSpacing()
shape = img.GetSize()
print(spacing)
print(shape)

a = load_dict_csv(valid_landmark_path)
print(a)



renderer = Renderer()
renderer.set_camera(pos=(0,-1000,0))
renderer.add_actor(VolumeActor(uncertainty, 'uncertainty_2'))
origin = - np.array(shape) * np.array(spacing) / 2.



renderer.add_actor(LandmarkActor(tuple(np.array((69,144,335))+origin), radius=4, rgb=(0,0,0)))
# renderer.add_actor(LandmarkActor((19,,202.39574), radius=5, rgb=(1,0,0)))
img_1 = renderer.render(bg=(1,1,1), rotate_angles=[0])[0]
print(img_1.shape)
img_2 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

img = np.concatenate([img_1,img_2],axis=1)
save_path = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\visualization'
cv2.imwrite(os.path.join(save_path, 'landmark_test.png'), img)
print(img.shape)


