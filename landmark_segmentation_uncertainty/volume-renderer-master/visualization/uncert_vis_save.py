import os
import cv2
import numpy as np
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor



result_path = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29\results'
save_path = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29'
save_path = os.path.join(save_path, 'visualization')
if not os.path.exists(save_path):
    os.makedirs(save_path)

sub_folders = os.listdir(result_path)[2:]

for case_id in sub_folders:

    print(case_id)
    
    label = os.path.join(r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\verse2019_dataset\images_reoriented',case_id+'_seg.nii.gz')
    sub_folder = os.path.join(r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29\results', case_id)
    pred = os.path.join(sub_folder, case_id+'_pred_seg.nii.gz')
    uncertainty = os.path.join(sub_folder, case_id+'_pred_uncert.nii.gz')
    error = os.path.join(sub_folder, case_id+'_error.nii.gz')

    renderer = Renderer()
    renderer.set_camera(pos=(0,-1400,0))
    renderer.add_actor(SurfaceActor(label, 'vertebrae'))
    img_1 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

    renderer = Renderer()
    renderer.set_camera(pos=(0,-1400,0))
    renderer.add_actor(SurfaceActor(pred, 'vertebrae'))
    img_2 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

    renderer = Renderer()
    renderer.set_camera(pos=(0,-1400,0))
    renderer.add_actor(VolumeActor(uncertainty, 'uncertainty_2'))
    img_3 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

    renderer = Renderer()
    renderer.set_camera(pos=(0,-1400,0))
    renderer.add_actor(VolumeActor(error, 'error_2'))
    img_4 = renderer.render(bg=(1,1,1), rotate_angles=[90])[0]

    img = img_1
    img = np.concatenate([img,img_2],axis=1)
    img = np.concatenate([img,img_3],axis=1)
    img = np.concatenate([img,img_4],axis=1)

    cv2.imwrite(os.path.join(save_path, case_id+'.png'), img)