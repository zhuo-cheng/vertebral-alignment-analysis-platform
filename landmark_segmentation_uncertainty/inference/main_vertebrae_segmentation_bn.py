#!/usr/bin/python
import sys
sys.path.append('/win/scallop/user/cheng/public_platform/v1_public/landmark_segmentation_uncertainty/MedicalDataAugmentationTool-master')

import time
import datetime
import argparse
from collections import OrderedDict
from glob import glob

import SimpleITK as sitk
import numpy as np
import os
import traceback
import tensorflow as tf

import utils.io.image
import utils.io.text
import utils.sitk_image
import utils.sitk_np
from dataset import Dataset
from network import network_u, UnetClassicAvgLinear3d
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.tensorflow_util import create_placeholders_tuple
from utils.segmentation.segmentation_test import SegmentationTest
from utils.sitk_np import np_to_sitk
import utils.np_image


class MainLoop(MainLoopBase):
    def __init__(self, network, unet, network_parameters, image_size, image_spacing, data_format):
        super().__init__()
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = data_format
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = image_size
        self.image_spacing = image_spacing

    def init_networks(self):
        network_image_size = list(reversed(self.image_size))

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('single_heatmap', [1] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('single_heatmap', network_image_size + [1])])

        data_generator_types = {'image': tf.float32, 'single_heatmap': tf.float32}


        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build val graph
        self.data_val, self.single_heatmap_val = create_placeholders_tuple(data_generator_entries, data_types=data_generator_types, shape_prefix=[1])
        concat_axis = 1 if self.data_format == 'channels_first' else 4
        self.data_heatmap_concat_val = tf.concat([self.data_val, self.single_heatmap_val], axis=concat_axis)
        self.prediction_val = training_net(self.data_heatmap_concat_val, num_labels=self.num_labels, is_training=True, actual_network=self.unet, padding=self.padding, data_format=self.data_format, **self.network_parameters)
        self.prediction_softmax_val = tf.nn.sigmoid(self.prediction_val)

    def test_full_image(self, image, heatmap):
        feed_dict = {self.data_val: np.expand_dims(image, axis=0),
                     self.single_heatmap_val: np.expand_dims(heatmap, axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_softmax_val,), feed_dict=feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)

        return prediction


class InferenceLoop(object):
    def __init__(self, network, unet, network_parameters, image_base_folder, setup_base_folder, load_model_filenames, output_base_folder):
        super().__init__()
        #self.load_model_filenames = ['/models/vertebrae_segmentation/model']
        #self.image_base_folder = '/tmp/data_reoriented'
        #self.setup_base_folder = '/tmp/'
        self.image_base_folder = image_base_folder
        self.setup_base_folder = setup_base_folder
        self.load_model_filenames = load_model_filenames
        self.num_labels = 1
        self.num_labels_all = 26
        self.data_format = 'channels_last'
        self.network = network
        self.unet = unet
        self.network_parameters = network_parameters
        self.padding = 'same'
        self.image_size = [128, 128, 96]
        self.image_spacing = [1] * 3
        self.save_debug_images = False
        self.uncert_cal_times = 10
        # self.output_folder = os.path.join(output_base_folder, 'result'+'_{}'.format(time_idx))
        self.output_folder = output_base_folder

        dataset_parameters = {'cv': 'inference',
                              'image_base_folder': self.image_base_folder,
                              'setup_base_folder': self.setup_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'input_gaussian_sigma': 0.75,
                              'label_gaussian_sigma': 1.0,
                              'heatmap_sigma': 3.0,
                              'generate_single_vertebrae_heatmap': True,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

        self.network_loop = MainLoop(network, unet, network_parameters, self.image_size, self.image_spacing, self.data_format)
        self.network_loop.init_networks()
        self.network_loop.init_variables()
        self.network_loop.init_saver()
        self.init_image_list()

    def init_image_list(self):
        images_files = sorted(glob(os.path.join(self.image_base_folder, '*.nii.gz')))
        self.image_id_list = map(lambda filename: os.path.basename(filename)[:-len('.nii.gz')], images_files)
        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'vertebrae_localization/valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file)

    def output_file_for_current_iteration(self, file_name):
        return os.path.join(self.output_folder, file_name)

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        predictions = []
        for load_model_filename in self.load_model_filenames:
            if len(self.load_model_filenames) > 1:
                self.network_loop.load_model_filename = load_model_filename
                self.network_loop.load_model()
            prediction = self.network_loop.test_full_image(generators['image'], generators['single_heatmap'])
            predictions.append(prediction)

        prediction = np.mean(predictions, axis=0)
        transformation = transformations['image']
        image = generators['image']

        return image, prediction, transformation

    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        if len(self.load_model_filenames) == 1:
            self.network_loop.load_model_filename = self.load_model_filenames[0]
            self.network_loop.load_model()


        labels = list(range(self.num_labels_all))
        interpolator = 'linear'
        filter_largest_cc = True
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator=interpolator,
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)


        filenames = os.listdir(self.image_base_folder)
        filenames = [i for i in filenames if i[-len('.nii.gz'):]]
        file_number = len(filenames)
        idx = 1

        # Create folder to save

        # Process each image
        for image_id in self.image_id_list:
            
            file_save_folder = os.path.join(self.output_folder, image_id)
            if not os.path.exists(file_save_folder):
                os.makedirs(file_save_folder)

            try:
                print("[{}/{}] Vertebrae Segmentation: {} ".format(idx, file_number, image_id))

                first = True
                uncert_prediction_resampled_np = None
                input_image = None

                for landmark_id in self.valid_landmarks[image_id]:

                    print('Landmark ID: {}'.format(landmark_id))

                    # Load patch input
                    dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id' : landmark_id})
                    datasources = dataset_entry['datasources']

                    # Create numpy array to save results with original size
                    if first:
                        input_image = datasources['image']
                        if self.data_format == 'channels_first':
                            uncert_prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                            uncert_prediction_resampled_np[0, ...] = 0.5
                            label_prediction_resampled_np = np.zeros([self.num_labels_all] + list(reversed(input_image.GetSize())), dtype=np.float16)
                            label_prediction_resampled_np[0, ...] = 0.5
                        else:
                            uncert_prediction_resampled_np = np.zeros(list(reversed(input_image.GetSize())) + [self.num_labels_all], dtype=np.float16)
                            label_prediction_resampled_np = np.zeros(list(reversed(input_image.GetSize())) + [self.num_labels_all], dtype=np.float16)
                            label_prediction_resampled_np[..., 0] = 0.5
                        first = False


                    # Calculate "self.uncert_cal_times" to utilize Bayesian CNN strategy
                    patch_cal_tmp_np = np.zeros( [self.uncert_cal_times] + list(reversed(self.image_size)) + [1])
                    for time_idx in range(self.uncert_cal_times):
                        image, prediction, transformation = self.test_full_image(dataset_entry)
                        patch_cal_tmp_np[time_idx,...] = prediction

                    patch_cal_label_np = np.mean(patch_cal_tmp_np, axis=0)
                    patch_cal_uncert_np = np.var(patch_cal_tmp_np, axis=0)
                        
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))

                    # Post-processing for eliminate small region
                    if filter_largest_cc:
                        prediction_thresh_np = (patch_cal_label_np > 0.5).astype(np.uint8)
                        if self.data_format == 'channels_first':
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[0])
                            prediction_thresh_np[largest_connected_component[None, ...] == 1] = 0
                        else:
                            largest_connected_component = utils.np_image.largest_connected_component(prediction_thresh_np[..., 0])
                            prediction_thresh_np[largest_connected_component[..., None] == 1] = 0
                        patch_cal_label_np[prediction_thresh_np == 1] = 0

                    if self.save_debug_images:
                        utils.io.image.write_multichannel_np(image, os.path.join(file_save_folder, image_id + '_' + landmark_id + '_input.mha'), output_normalization_mode='min_max', data_format=self.data_format, image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(patch_cal_label_np, os.path.join(file_save_folder, image_id + '_' + landmark_id + '_prediction.mha'), data_format=self.data_format, image_type=np.double, spacing=self.image_spacing, origin=origin)
                        utils.io.image.write_multichannel_np(patch_cal_uncert_np, os.path.join(file_save_folder, image_id + '_' + landmark_id + 'uncert.mha'), data_format=self.data_format, image_type=np.double, spacing=self.image_spacing, origin=origin)

                    # Segmentation mask calculation processing
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=patch_cal_label_np,
                                                                                        output_spacing=self.image_spacing,
                                                                                        channel_axis=channel_axis,
                                                                                        input_image_sitk=input_image,
                                                                                        transform=transformation,
                                                                                        interpolator=interpolator,
                                                                                        output_pixel_type=sitk.sitkFloat32)
                    if self.data_format == 'channels_first':
                        label_prediction_resampled_np[int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    else:
                        label_prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    
                    # Uncertainty calculation processing
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=patch_cal_uncert_np,
                                                                                        output_spacing=self.image_spacing,
                                                                                        channel_axis=channel_axis,
                                                                                        input_image_sitk=input_image,
                                                                                        transform=transformation,
                                                                                        interpolator=interpolator,
                                                                                        output_pixel_type=sitk.sitkFloat32)
                    if self.data_format == 'channels_first':
                        uncert_prediction_resampled_np[int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    else:
                        uncert_prediction_resampled_np[..., int(landmark_id) + 1] = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])


                prediction_labels = segmentation_test.get_label_image(label_prediction_resampled_np, reference_sitk=input_image, image_type=np.uint16)
                utils.io.image.write(prediction_labels, os.path.join(file_save_folder, image_id + '_pred_seg.nii.gz'))

                prediction_uncert_np = np.mean(uncert_prediction_resampled_np, axis=-1)     
                prediction_labels_sitk = np_to_sitk(prediction_uncert_np, type=np.float32)
                prediction_labels_sitk.CopyInformation(input_image)
                sitk.WriteImage(prediction_labels_sitk,os.path.join(file_save_folder, image_id + '_pred_uncert.nii.gz'))

                idx += 1

            except:
                idx += 1
                print(traceback.format_exc())
                print('ERROR predicting', image_id)
                pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--setup_folder', type=str, required=True)
    parser.add_argument('--model_files', nargs='+', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser_args = parser.parse_args()
    network_parameters = OrderedDict([('num_filters_base', 64), ('double_features_per_level', False), ('num_levels', 5), ('activation', 'relu')])

    # Create dir for saving multiple results
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = 'vertebrae_bayesian_segmentation_rai'
        
    loop = InferenceLoop(network_u, UnetClassicAvgLinear3d, network_parameters, parser_args.image_folder, parser_args.setup_folder, parser_args.model_files, os.path.join(parser_args.output_folder, result_folder))
    loop.test()
    tf.reset_default_graph()