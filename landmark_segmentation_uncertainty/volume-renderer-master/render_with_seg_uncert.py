import os
import cv2
import itk
import csv
import vtk
import numpy as np
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor
from pyvr.actors import VertebraVolumeActor


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented
    
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


def load_each_ver_coor(valid_landmarks, landmarks_coor):

    all_vert_list = []

    for idx in valid_landmarks:
        one_vert_list = []
        try:
            idx = int(idx)
            one_vert_list.append(idx)
            one_vert_coor = np.array((float(landmarks_coor[idx*3]),float(landmarks_coor[idx*3+1]),float(landmarks_coor[idx*3+2])))
            one_vert_list.append(one_vert_coor)

            # print(one_vert_list)
            all_vert_list.append(one_vert_list)

        except:
            continue

    return all_vert_list
    


if __name__ == "__main__":

    img_folder = r'\\scallop\User\cheng\public_platform\v1_public\test\img'
    seg_folder = r'\\scallop\User\cheng\public_platform\v1_public\test\results\vertebrae_bayesian_segmentation_original'
    vis_folder = r'\\scallop\User\cheng\public_platform\v1_public\test\visualization'

    save_folder = os.path.join(vis_folder, 'segmentation_uncertainty')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    sub_folders = os.listdir(seg_folder)
    file_number = len(sub_folders)
    idx = 1

    for case_id in sub_folders:

        print("[{}/{}] Visualization of segmentation and uncertainty: {}".format(idx, file_number, case_id))
        idx += 1
        
        img_path = os.path.join(img_folder, case_id + '.nii.gz')
        seg_path = os.path.join(seg_folder, case_id, case_id + '_pred_seg.nii.gz') 
        uncert_path = os.path.join(seg_folder, case_id, case_id + '_pred_uncert.nii.gz') 

        # Reference to RAI 
        img = itk.imread(img_path, itk.F)
        img_reoriented = reorient_to_rai(img)
        img_reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        img_reoriented.SetDirection(m)
        img_reoriented.Update()
        img_reoriented_np = itk.GetArrayFromImage(img_reoriented)

        # Reference to RAI (segmentation)
        seg = itk.imread(seg_path, itk.SS)
        seg_reoriented = reorient_to_rai(seg)
        seg_reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        seg_reoriented.SetDirection(m)
        seg_reoriented.Update()
        seg_reoriented_np = itk.GetArrayFromImage(seg_reoriented)

        # Reference to RAI (uncertainty)
        uncert = itk.imread(uncert_path, itk.F)
        uncert_reoriented = reorient_to_rai(uncert)
        uncert_reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        uncert_reoriented.SetDirection(m)
        uncert_reoriented.Update()
        uncert_reoriented_np = itk.GetArrayFromImage(uncert_reoriented)

        # Load info of RAI image
        reorient_size = np.flip(img_reoriented.shape)
        reorient_spacing = tuple(img_reoriented.GetSpacing())
        reorient_direction = tuple(itk.GetArrayFromMatrix(img_reoriented.GetDirection()))

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-2.2*camera_locat,0))

        # Segmentation rendering
        renderer.add_actor(VertebraVolumeActor(img_reoriented_np, spacing=reorient_spacing))
        renderer.add_actor(SurfaceActor(seg_reoriented_np, spacing=reorient_spacing))
        seg_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[180])[0]
        seg_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-2.2*camera_locat,0))

        # Uncertainty rendering
        renderer.add_actor(VolumeActor(uncert_reoriented_np, spacing=reorient_spacing, preset='vert_uncertainty'))
        uncert_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[180])[0]
        uncert_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]
        
        # Save the image
        save_fig = np.concatenate([seg_np_0, uncert_np_0, seg_np_90, uncert_np_90], axis=1)
        save_filepath = os.path.join(save_folder, case_id+'.png')
        cv2.imwrite(save_filepath, save_fig)