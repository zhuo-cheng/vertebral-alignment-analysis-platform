import os
import cv2
import itk
import csv
import vtk
import argparse
import numpy as np
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor
from pyvr.actors import VertebraVolumeActor
from pyvr.actors import LandmarkActor

landmark_colormap = {
        'label_0': [0,0,0], 'label_1':[1,0,0], 'label_2':[0,1,0],
        'label_3':[0,0,1], 'label_4':[1,1,0], 'label_5' : [0,1,1],
        'label_6' : [1,0,1], 'label_7':[1,0.937,0.835], 'label_8':[0,0,0.804],
        'label_9':[0.804,0.522,0.247], 'label_10':[0.824,0.706,0.549], 'label_11':[0.4,0.804,0.667],
        'label_12':[0,0,0.5], 'label_13':[0,0.545,0.545], 'label_14':[0.180,0.545,0.341],
        'label_15':[1,0.894,0.882], 'label_16':[0.416,0.353,0.804], 'label_17':[0.86,0.627,0.86],
        'label_18':[0.914,0.588,0.478], 'label_19':[0.647,0.1647,0.1647], 'label_20':[1,0.98,0.98],
        'label_21':[0.576,0.439,0.859], 'label_22':[0.855,0.439,0.839], 'label_23':[0.294,0,0.510],
        'label_24':[1,0.714,0.757], 'label_25':[0.235,0.702,0.443]
    }

def choose_pos_value(label_num):

    if label_num <= 6:
        pos_value = 4
    elif label_num <= 13:
        pos_value = 3
    else:
        pos_value = 2.2

    return pos_value

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--seg_folder', type=str, required=True)
    parser.add_argument('--csv_folder', type=str, required=True)
    parser.add_argument('--vis_folder', type=str, required=True)
    parser_args = parser.parse_args()

    img_folder = parser_args.img_folder
    seg_folder = parser_args.seg_folder
    csv_folder = parser_args.csv_folder
    vis_folder = parser_args.vis_folder

    save_folder = os.path.join(vis_folder, 'all_results_vis')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load landmark csv files
    valid_landmarks = load_dict_csv(os.path.join(csv_folder, 'valid_landmarks.csv'))
    landmarks_coor = load_dict_csv(os.path.join(csv_folder, 'landmarks.csv'))

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

        # Set camera according the number of vertebrae
        label_num = len(np.unique(seg_reoriented_np)) - 1
        pos_value = choose_pos_value(label_num)

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-pos_value*camera_locat,0))

        # CT volume rendering
        renderer.add_actor(VertebraVolumeActor(img_reoriented_np, spacing=reorient_spacing))

        # Load coordinates of each landmark
        coor_list = load_each_ver_coor(valid_landmarks[case_id], landmarks_coor[case_id])

        # Add landmarks to the rendered CT volume
        origin = - np.array(reorient_size) * np.array(reorient_spacing) / 2.
        for one_ver_coor in coor_list:
            label_colormap_keyname = 'label_' + str(one_ver_coor[0]+1)
            label_color = tuple(landmark_colormap[label_colormap_keyname])
            renderer.add_actor(LandmarkActor((tuple(one_ver_coor[1])+origin), radius=4, rgb=label_color))
        proj_0 = renderer.render(rotate_angles=[180], bg=(1,1,1))[0]
        proj_90 = renderer.render(rotate_angles=[90], bg=(1,1,1))[0]

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-pos_value*camera_locat,0))

        # Segmentation rendering
        renderer.add_actor(VertebraVolumeActor(img_reoriented_np, spacing=reorient_spacing))
        renderer.add_actor(SurfaceActor(seg_reoriented_np, spacing=reorient_spacing))
        seg_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[180])[0]
        seg_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-pos_value*camera_locat,0))

        # Uncertainty rendering
        renderer.add_actor(VolumeActor(uncert_reoriented_np, spacing=reorient_spacing, preset='vert_uncertainty'))
        uncert_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[180])[0]
        uncert_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]
        
        # Save the image
        save_fig_back = np.concatenate([proj_0, seg_np_0, uncert_np_0], axis=1)
        save_fig_side = np.concatenate([proj_90, seg_np_90, uncert_np_90], axis=1)
        save_fig = np.concatenate([save_fig_back, save_fig_side], axis=0)
        save_filepath = os.path.join(save_folder, case_id+'.png')
        cv2.imwrite(save_filepath, save_fig)