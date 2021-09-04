import argparse
import itk
from glob import glob
import os
import numpy as np

def reorient_to_reference(image, reference):
    """
    Reorient image to reference. See itk.OrientImageFilter.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image reoriented to reference image.
    """
    filter = itk.OrientImageFilter[type(image), type(image)].New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    filter.SetDesiredCoordinateDirection(reference.GetDirection())
    filter.Update()
    return filter.GetOutput()


def change_image_type(old_image, new_type):
    """
    Cast image to reference image type.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image cast to reference image type.
    """
    filter = itk.CastImageFilter[type(image), new_type].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()

def cast(image, reference):
    """
    Cast image to reference image type.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image cast to reference image type.
    """
    filter = itk.CastImageFilter[type(image), type(reference)].New()
    filter.SetInput(image)
    filter.Update()
    return filter.GetOutput()


def copy_information(image, reference):
    """
    Copy image information (spacing, origin, direction) from reference image to input image.
    :param image: Input itk image.
    :param reference: Reference itk image.
    :return: Input image with image information from reference image.
    """
    filter = itk.ChangeInformationImageFilter[type(image)].New()
    filter.SetInput(image)
    filter.SetReferenceImage(reference)
    filter.UseReferenceImageOn()
    filter.ChangeSpacingOn()
    filter.ChangeOriginOn()
    filter.ChangeDirectionOn()
    filter.Update()
    return filter.GetOutput()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--reference_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()

    if not os.path.exists(parser_args.output_folder):
        os.makedirs(parser_args.output_folder)

    sub_folders = os.listdir(parser_args.image_folder)
    
    file_number = len(sub_folders)
    idx = 1

    for sub_folder in sorted(sub_folders):
        
        print("[{}/{}] RAI to Reference: {}".format(idx, file_number, sub_folder))
        idx += 1

        sub_folder_path = os.path.join(parser_args.image_folder, sub_folder)
        filenames = os.listdir(sub_folder_path)

        for filename in sorted(filenames):
            basename = os.path.basename(filename)
            img_path = os.path.join(sub_folder_path,filename)
            image = itk.imread(img_path)

            reference = itk.imread(os.path.join(parser_args.reference_folder, sub_folder + '.nii.gz'))
            reference = change_image_type(reference, itk.Image[itk.F,3])
            reoriented = cast(image, reference)

            reoriented = reorient_to_reference(reoriented, reference)
            reoriented = copy_information(reoriented, reference)
            save_sub_folder = os.path.join(parser_args.output_folder, sub_folder)
            if not os.path.exists(save_sub_folder):
                os.makedirs(save_sub_folder)
            itk.imwrite(reoriented, os.path.join(save_sub_folder, filename + '.nii.gz'))
