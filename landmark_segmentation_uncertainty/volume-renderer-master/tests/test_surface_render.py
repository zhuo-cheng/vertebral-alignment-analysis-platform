from pyvr import surface_render
from pyvr.utils.video import write_video

if __name__ == '__main__':

    volume = r'\\scallop\User\cheng\Project\spine_segmentation\experiments\verse_seg_exp5_BS\MedicalDataAugmentationTool-VerSe-master\stat_model\process\eval\results\500_true_500_false_eval\vis_test\img\17035395872_1210161042308715_4.nii.gz'
    preset = 'bone'

    proj = surface_render(volume, preset, pos=(0,-1200,0))
    write_video(proj, 'test.mp4')
