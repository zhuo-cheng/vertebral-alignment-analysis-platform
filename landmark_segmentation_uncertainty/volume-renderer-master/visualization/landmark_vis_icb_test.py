import os
import csv
import cv2
import argparse
import numpy as np
import traceback
import SimpleITK as sitk
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor
from pyvr.actors import LandmarkActor


label_colormap = {
        'label_0': [0,0,0], 'label_1':[0,0,255], 'label_2':[0,255,0],
        'label_3':[255,0,0], 'label_4':[0,255,255], 'label_5' : [255,255,0],
        'label_6' : [255,0,255], 'label_7':[213,239,255], 'label_8':[205,0,0],
        'label_9':[63,133,205], 'label_10':[140,180,210], 'label_11':[170,205,102],
        'label_12':[128,0,0], 'label_13':[139,139,0], 'label_14':[87,139,46],
        'label_15':[225,228,255], 'label_16':[205,90,106], 'label_17':[221,160,221],
        'label_18':[122,150,233], 'label_19':[42,42,165], 'label_20':[250,250,255],
        'label_21':[219,112,147], 'label_22':[214,112,218], 'label_23':[130,0,75],
        'label_24':[193,182,255], 'label_25':[113,179,60]
    }

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
        
    # print(all_vert_list)
    return all_vert_list
    
    

def to_greyscale(slice_data):
    rows,cols = slice_data.shape
    # print(rows,cols)
    max = -1024
    min = 5000

    for i in range(rows):
        for j in range(cols):
            if slice_data[i,j] > max:
                max = slice_data[i,j]
            
            if slice_data[i,j] < min:
                min = slice_data[i,j]

    k = 255/(max-min)
    b = 255-(k*max)
    
    for i in range(rows):
        for j in range(cols):
            slice_data[i,j] = k*slice_data[i,j]+b

    return slice_data


def img_scale(img, width=None, height=None):

    if not width and not height:
        width, height = img.shape
    if not width or not height:
        _width, _height = img.shape
        height = width * _height / _width if width else height
        width = height * _width / _height if height else width
    return width, height



def choose_pos_value(vertebrae_num):

    if vertebrae_num <= 3:
        pos_value = -350
    elif vertebrae_num <= 7:
        pos_value = -500
    if vertebrae_num <= 12:
        pos_value = -650
    elif vertebrae_num <=17:
        pos_value = -850
    elif vertebrae_num <= 22:
        pos_value = -1100
    else:
        pos_value = -1300

    return pos_value


def get_color_seg_mask(CT_slice, seg_slice):

    CT_slice = np.asarray(CT_slice, dtype=np.uint8) 
    CT_slice_BGR = cv2.cvtColor(CT_slice,cv2.COLOR_GRAY2BGR)
    # print(CT_slice_BGR.shape)

    rows,cols,channels = CT_slice_BGR.shape
    # print(rows,cols)
    for i in range(rows):
        for j in range(cols):
            if seg_slice[i,j] != 0:
                label_value = seg_slice[i,j]
                CT_slice_BGR[i,j] = label_colormap['label_' + str(label_value)]
    return CT_slice_BGR


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', type=str, required=True)
    # parser.add_argument('--seg_result_folder', type=str, required=True)
    # parser.add_argument('--landmark_result_folder', type=str, required=True)
    # parser.add_argument('--save_folder', type=str, required=True)
    # parser_args = parser.parse_args()
    # save_folder = parser_args.save_folder
    
    

    image_folder = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\img_reoriented'
    seg_result_folder = r'\\scallop\User\cheng\for_Otake_sensei\21_02_17_test_results\vertebrae_segmentation_bayesian_reorient'
    landmark_result_folder = r'\\scallop\User\Nara Medic Univ Orthopaedic\Spine\20210216_preliminary_test\21_02_17_test_results\vertebrae_localization'
    save_folder = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\test_for_new_case\20210217_NaraMedU\visualization' 


    # New a folder to save results  
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Get the sub folders of segmentation and uncertainty
    sub_folders = os.listdir(seg_result_folder)
    
    # Load landmark csv files
    valid_landmarks = load_dict_csv(os.path.join(landmark_result_folder, 'valid_landmarks.csv'))
    landmarks_coor = load_dict_csv(os.path.join(landmark_result_folder, 'landmarks.csv'))


    for case_id in sub_folders:

        try:

            print(case_id)

            # Get segmentation mask and uncertainty file paths
            sub_folder = os.path.join(seg_result_folder,case_id)
            pred_path = os.path.join(sub_folder, case_id+'_pred_seg.nii.gz')
            uncertainty_path = os.path.join(sub_folder, case_id+'_pred_uncert.nii.gz')


            # load segmentation mask image
            prediction = sitk.ReadImage(pred_path)
            prediction_np = sitk.GetArrayFromImage(prediction)


            # Calculate slice idx for visualization
            # slice_idx = int(prediction_np.shape[-1] / 2)
            choosen_slice_spine_idx = np.nonzero(prediction_np != 0)
            slice_idx = int((choosen_slice_spine_idx[-1][0] + choosen_slice_spine_idx[-1][-1]) / 2)


            # load CT image
            img_path = os.path.join(image_folder, case_id+'.nii.gz')
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)
            # slice_idx = int(img_np.shape[-1] / 2)


            # Load CT image slice and segmentation mask slice
            img_vis_slice = img_np[...,slice_idx]
            img_vis_slice = np.flipud(img_vis_slice)
            img_vis_slice_norm = to_greyscale(img_vis_slice)
            pred_vis_slice = prediction_np[...,slice_idx]
            pred_vis_slice = np.flipud(pred_vis_slice)
            pred_vis_slice_mer = get_color_seg_mask(img_vis_slice_norm, pred_vis_slice)


            # Resize considering original spacing
            original_spacing = img.GetSpacing()
            original_size = img.GetSize()
            spacing_new_size = (int(original_spacing[1]*original_size[1]), int(original_spacing[-1]*original_size[-1]))
            # print(spacing_new_size)
            img_vis_slice_norm = cv2.resize(img_vis_slice_norm, spacing_new_size)
            pred_vis_slice_mer = cv2.resize(pred_vis_slice_mer, spacing_new_size)


            # Rescale image for visualization
            width, height = img_scale(img_vis_slice_norm, width = 800)
            before_pad_new_size = (int(height), int(width))
            img_vis_slice_norm = cv2.resize(img_vis_slice_norm, before_pad_new_size)
            pred_vis_slice_mer = cv2.resize(pred_vis_slice_mer, before_pad_new_size)


            # Calculate voxel to move camera to localize spine to mid in the image
            try:
                pred_vis_slice_spine_idx = np.nonzero(pred_vis_slice != 0)
            except:
                pred_vis_slice_spine_idx = ([0],[0])

            spine_median_idx_ud = int((pred_vis_slice_spine_idx[0][0] + pred_vis_slice_spine_idx[0][-1]) / 2)
            move_voxel_num_ud = int(pred_vis_slice.shape[0] / 2) - spine_median_idx_ud
            spine_median_idx_lr = int((pred_vis_slice_spine_idx[1][0] + pred_vis_slice_spine_idx[1][-1]) / 2)
            move_voxel_num_lr = spine_median_idx_lr - int(pred_vis_slice.shape[1] / 2)

            
            # Calculate the number of vertebrae to large camera view
            vertebrae_num = len(np.unique(prediction_np)) - 1
            pos_value = choose_pos_value(vertebrae_num)


            # Render the volume for segmentation
            renderer = Renderer()
            renderer.set_camera(pos=(0,pos_value,0), fp=(0,move_voxel_num_lr,move_voxel_num_ud))
            renderer.add_actor(SurfaceActor(pred_path, 'vertebrae'))
            pred_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[0])[0]
            pred_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

            # Render the volume for uncertainty
            renderer = Renderer()
            renderer.set_camera(pos=(0,pos_value,0), fp=(0,move_voxel_num_lr,move_voxel_num_ud))
            renderer.add_actor(VolumeActor(uncertainty_path, 'uncertainty_2'))

            # Add landmarks to the volume
            coor_list = load_each_ver_coor(valid_landmarks[case_id], landmarks_coor[case_id])
            
            origin = - np.array(original_size) * np.array(original_spacing) / 2.
            for one_ver_coor in coor_list:
                label_colormap_keyname = 'label_' + str(one_ver_coor[0]+1)
                label_color = tuple(landmark_colormap[label_colormap_keyname])

                renderer.add_actor(LandmarkActor((tuple(one_ver_coor[1])+origin), radius=4, rgb=label_color))

            uncert_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[0])[0]
            uncert_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]


            # Resize
            if img_vis_slice.shape[0] >= 800:
                width, height = img_scale(img_vis_slice_norm, width = 800)
                new_size = (int(height), int(width))
                img_vis_slice_resized = cv2.resize(img_vis_slice_norm, new_size)
                pred_vis_slice_resized = cv2.resize(pred_vis_slice_mer, new_size)

            else:
                pad_border_1 = int((800-img_vis_slice_norm.shape[0])/2)
                pad_border_2 = 800-pad_border_1-img_vis_slice_norm.shape[0]
                img_vis_slice_resized = np.pad(img_vis_slice_norm, ((pad_border_1,pad_border_2),(0,0)), 'constant') 
                pred_vis_slice_resized = np.pad(pred_vis_slice_mer, ((pad_border_1,pad_border_2),(0,0),(0,0)), 'constant') 
                # new_size = (img_vis_slice_resized.shape[1], img_vis_slice_resized.shape[0])


            img_vis_slice_resized = cv2.cvtColor(np.asarray(img_vis_slice_resized, dtype=np.uint8),cv2.COLOR_GRAY2BGR)

            # print(img_vis_slice_resized.shape)
            # print(pred_vis_slice_resized.shape)
            # print(pred_np_90.shape)
            # print(uncert_np_90.shape)
            # print(pred_np_0.shape)
            # print(uncert_np_0.shape)

            img = np.concatenate([img_vis_slice_resized, pred_vis_slice_resized, pred_np_90, uncert_np_90, pred_np_0, uncert_np_0], axis=1)
            cv2.imwrite(os.path.join(save_folder, case_id+'.png'), img)

        except:
                      
            print(traceback.format_exc())
            print('ERROR:', case_id)
            pass