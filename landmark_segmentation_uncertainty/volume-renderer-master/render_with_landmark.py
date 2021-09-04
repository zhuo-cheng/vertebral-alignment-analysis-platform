import os
import cv2
import itk
import csv
import vtk
import numpy as np
from pyvr.renderer import Renderer
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
    csv_folder = r'\\scallop\User\cheng\public_platform\v1_public\test\results\vertebrae_localization'
    vis_folder = r'\\scallop\User\cheng\public_platform\v1_public\test\visualization'

    save_folder = os.path.join(vis_folder, 'landmarks')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load landmark csv files
    valid_landmarks = load_dict_csv(os.path.join(csv_folder, 'valid_landmarks.csv'))
    landmarks_coor = load_dict_csv(os.path.join(csv_folder, 'landmarks.csv'))

    filenames = os.listdir(img_folder)
    file_number = len(filenames)
    idx = 1

    for filename in filenames:

        case_id = filename.split('.')[0]
        img_path = os.path.join(img_folder, filename)
        print("[{}/{}] Volume rendering with landmarks: {}".format(idx, file_number, case_id))
        idx += 1

        # Reference to RAI 
        image = itk.imread(img_path, itk.SS)
        reoriented = reorient_to_rai(image)
        
        reoriented.SetOrigin([0, 0, 0])
        m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        reoriented.SetDirection(m)
        reoriented.Update()

        # Load info of RAI image
        image_np = itk.GetArrayFromImage(reoriented)
        reorient_size = np.flip(image_np.shape)
        reorient_spacing = tuple(reoriented.GetSpacing())
        reorient_direction = tuple(itk.GetArrayFromMatrix(reoriented.GetDirection()))

        # Set camera localization
        renderer = Renderer()
        camera_locat = reorient_size[-1] * reorient_spacing[-1]
        renderer.set_camera(pos=(0,-2.2*camera_locat,0))

        # CT volume rendering
        renderer.add_actor(VertebraVolumeActor(image_np, reorient_spacing))

        # Load coordinates of each landmark
        coor_list = load_each_ver_coor(valid_landmarks[case_id], landmarks_coor[case_id])

        # Add landmarks to the rendered CT volume
        origin = - np.array(reorient_size) * np.array(reorient_spacing) / 2.
        for one_ver_coor in coor_list:
            label_colormap_keyname = 'label_' + str(one_ver_coor[0]+1)
            label_color = tuple(landmark_colormap[label_colormap_keyname])
            renderer.add_actor(LandmarkActor((tuple(one_ver_coor[1])+origin), radius=4, rgb=label_color))

        # Adjust the orientation
        proj_0 = renderer.render(rotate_angles=[180], bg=(1,1,1))[0]
        proj_90 = renderer.render(rotate_angles=[90], bg=(1,1,1))[0]

        # Save the image
        save_fig = np.concatenate([proj_0,proj_90], axis=1)
        save_filepath = os.path.join(save_folder, case_id+'.png')
        cv2.imwrite(save_filepath, save_fig)

