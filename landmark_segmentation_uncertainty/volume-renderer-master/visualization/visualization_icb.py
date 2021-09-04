import os
import cv2
import argparse
import numpy as np
import traceback
import SimpleITK as sitk
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor


'''
CT slice
CT slice with segmentation
3D segmentation volume 0 degree
3D uncertainty volume 0 degree (landmark)
3D segmentation volume 90 degree
3D uncertainty volume 90 degree (landmark)
'''


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


def get_color_seg_mask(CT_slice, seg_slice):
    label = {
        'label_0': [0,0,0], 'label_1':[0,0,255], 'label_2':[0,255,0],
        'label_3':[255,0,0], 'label_4':[0,255,255], 'label_5' : [255,255,0],
        'label_6' : [255,0,255], 'label_7':[213,239,255], 'label_8':[205,0,0],
        'label_9':[63,133,205], 'label_10':[140,180,210], 'label_11':[170,205,102],
        'label_12':[0,0,128], 'label_13':[0,139,139], 'label_14':[87,139,46],
        'label_15':[225,228,255], 'label_16':[205,90,106], 'label_17':[221,160,221],
        'label_18':[122,150,233], 'label_19':[42,42,165], 'label_20':[250,250,255],
        'label_21':[219,112,147], 'label_22':[214,112,218], 'label_23':[130,0,75],
        'label_24':[193,182,255], 'label_25':[113,179,60]
    }

    CT_slice = np.asarray(CT_slice, dtype=np.uint8) 
    CT_slice_BGR = cv2.cvtColor(CT_slice,cv2.COLOR_GRAY2BGR)
    # print(CT_slice_BGR.shape)

    rows,cols,channels = CT_slice_BGR.shape
    # print(rows,cols)
    for i in range(rows):
        for j in range(cols):
            if seg_slice[i,j] != 0:
                label_value = seg_slice[i,j]
                CT_slice_BGR[i,j] = label['label_' + str(label_value)]
    return CT_slice_BGR


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', type=str, required=True)
    # parser.add_argument('--result_folder', type=str, required=True)
    # parser.add_argument('--save_folder', type=str, required=True)
    # parser_args = parser.parse_args()
    # save_folder = parser_args.save_folder

    image_folder = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\verse2019_dataset\images_reoriented'
    result_folder = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\final_all_results_21_01_29\results\vertebrae_segmentation_bayesian'
    save_folder = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\final_all_results_21_01_29\21_05_05_vis' 
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    sub_folders = os.listdir(result_folder)
    

    for case_id in sub_folders:

        try:

            print(case_id)

            sub_folder = os.path.join(result_folder,case_id)
            pred_path = os.path.join(sub_folder, case_id+'_pred_seg.nii.gz')
            uncertainty_path = os.path.join(sub_folder, case_id+'_pred_uncert.nii.gz')

            prediction = sitk.ReadImage(pred_path)
            prediction_np = sitk.GetArrayFromImage(prediction)
            choosen_slice_spine_idx = np.nonzero(prediction_np != 0)
            # slice_idx = int(prediction_np.shape[-1] / 2)
            slice_idx = int((choosen_slice_spine_idx[-1][0] + choosen_slice_spine_idx[-1][-1]) / 2)

            # CT image (slice)
            img_path = os.path.join(image_folder, case_id+'.nii.gz')
    
            img = sitk.ReadImage(img_path)
            original_spaing = img.GetSpacing()
            original_size = img.GetSize()

            print(original_spaing)
            print(original_size)

            img_np = sitk.GetArrayFromImage(img)
            # slice_idx = int(img_np.shape[-1] / 2)
            img_vis_slice = img_np[...,slice_idx]
            img_vis_slice = np.flipud(img_vis_slice)
            img_vis_slice_norm = to_greyscale(img_vis_slice)


            # segmentation image (slice)
            pred_vis_slice = prediction_np[...,slice_idx]
            pred_vis_slice = np.flipud(pred_vis_slice)
            pred_vis_slice_mer = get_color_seg_mask(img_vis_slice_norm, pred_vis_slice)

            # resize considering original spacing

            spacing_new_size = (int(original_spaing[1]*original_size[1]), int(original_spaing[-1]*original_size[-1]))
            print(spacing_new_size)
            img_vis_slice_norm = cv2.resize(img_vis_slice_norm, spacing_new_size)
            pred_vis_slice_mer = cv2.resize(pred_vis_slice_mer, spacing_new_size)

            width, height = img_scale(img_vis_slice_norm, width = 800)
            before_pad_new_size = (int(height), int(width))

            print(pred_vis_slice_mer.shape)
            print(img_vis_slice_norm.shape)

            img_vis_slice_norm = cv2.resize(img_vis_slice_norm, before_pad_new_size)
            pred_vis_slice_mer = cv2.resize(pred_vis_slice_mer, before_pad_new_size)

            print(pred_vis_slice_mer.shape)
            print(img_vis_slice_norm.shape)


            # calculate voxel to move camera to localize spine to mid in the image
            try:
                pred_vis_slice_spine_idx = np.nonzero(pred_vis_slice != 0)
            except:
                pred_vis_slice_spine_idx = ([0],[0])


            spine_median_idx_ud = int((pred_vis_slice_spine_idx[0][0] + pred_vis_slice_spine_idx[0][-1]) / 2)
            move_voxel_num_ud = int(pred_vis_slice.shape[0] / 2) - spine_median_idx_ud
            spine_median_idx_lr = int((pred_vis_slice_spine_idx[1][0] + pred_vis_slice_spine_idx[1][-1]) / 2)
            move_voxel_num_lr = spine_median_idx_lr - int(pred_vis_slice.shape[1] / 2)



            # calculate the number of vertebrae to large camera view
            vertebrae_num = len(np.unique(prediction_np)) - 1
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


            # prediction (volume)
            renderer = Renderer()
            # renderer.set_camera(pos=(0,pos_value,0), fp=(move_voxel_num_lr,0,move_voxel_num_ud))
            renderer.set_camera(pos=(0,pos_value,0), fp=(0,move_voxel_num_lr,move_voxel_num_ud))
            renderer.add_actor(SurfaceActor(pred_path, 'vertebrae'))
            pred_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[0])[0]
            pred_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

            # uncertainty (volume)
            renderer = Renderer()
            renderer.set_camera(pos=(0,pos_value,0), fp=(0,move_voxel_num_lr,move_voxel_num_ud))
            # renderer.set_camera(pos=(0,pos_value,0), fp=(0,0,0))

            renderer.add_actor(VolumeActor(uncertainty_path, 'uncertainty_2'))
            uncert_np_0 = renderer.render(bg=(1,1,1), rotate_angles=[0])[0]
            uncert_np_90 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]


            # resize
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