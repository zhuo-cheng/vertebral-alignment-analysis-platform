from pyvr.renderer import InteractiveMultiViewRenderer
from pyvr.renderer import Renderer
from pyvr.actors import VolumeActor
from pyvr.actors import SurfaceActor

label = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\verse2019_dataset\images_reoriented\verse004_seg.nii.gz'
pred = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29\results\verse004\verse004_pred_seg.nii.gz'
uncertainty = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29\results\verse004\verse004_pred_uncert.nii.gz'
error = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\verse2019\results\6_folds_eval\2020_11_24\final_all_results_21_01_29\results\verse004\verse004_error.nii.gz'



renderer = InteractiveMultiViewRenderer()
renderer.set_camera(pos=(0,-1200,0))
renderer.add_actor(SurfaceActor(label, 'vertebrae'))
renderer.add_actor(SurfaceActor(pred, 'vertebrae'))
renderer.add_actor(VolumeActor(uncertainty, 'uncertainty_2'))
renderer.add_actor(LandmarkActor((99.658,-53.036,-195.258), 10, rgb=(1,0,0)))
renderer.add_actor(LandmarkActor((-105.237,-57.957,-188.071), 10, rgb=(0,1,0)))
renderer.add_actor(LandmarkActor((0.753,-52.335,-105.167), 10, rgb=(0,0,1)))
# renderer.add_actor(SurfaceActor(error, 'vertebrae'))
renderer.add_actor(VolumeActor(error, 'error_2'))
renderer.render(bg=(1,1,1))

