import pandas as pd
import numpy as np
from glob import glob
import os
from os.path import join
import re
from PIL import Image
import seaborn as sns
from random import randrange
import pydicom
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
from skimage.transform import resize
import cv2

class Preprocess:

    # initialize a newly created object
    def __init__(self, path):
        self.path = path


    @classmethod
    def binarize(cls, value):
        """
        Convert a decimal value into a binary representation
        The length of the binary representation is limited into a size of 6
        since it only deals with a number [0 ... 63]
        """
        bin_list = [int(i) for i in bin(value)[2:]]
        bin_list.reverse()
        bin_list.extend([0] * (6 - len(bin_list)))
        return bin_list


    @classmethod
    def categorize(cls, label_list):
        train_label = []
        for i in range(0, len(label_list), 6):
            temp = label_list[i:i+6]
            sum = 0
            for j in range(0, len(temp)):
                sum = sum + temp[j] * 2**j
            train_label.append(sum)
        return train_label

        
    @classmethod
    def correct_dcm(cls, dcm):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x>=px_mode] = x[x>=px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000


    def decimalize(self, image_HU, window_width):
        """
        Convert values of Housefield Units into a range of (0, 1)
        image_HU: an image in Hounsefield Unit that outliers removed
        """
        min = np.amin(image_HU)
        # max = np.amax(image_HU)
        array_HU_shifted = image_HU - min  # array_HU is shifted by min

        array_HU_decimalized = array_HU_shifted / window_width
        return array_HU_decimalized


    def display_img_HU(self, dcm, window_center, window_width, intercept, slope):
        """
        displays a image transformed into Hounsefield Units from pixel values
        """
        # pixel_img = dcm.pixel_array
        HU_img = self.window_image(dcm, window_center,
                              window_width, intercept, slope)
        plt.imshow(HU_img, cmap=plt.cm.bone)
        plt.grid(False)


    @classmethod
    def get_datas(cls, dicom_file_lst):
        """
        Return a list of DICOM headers(Metadatas) and pixel_arryas from the DCM file list
        """
        dcm_metadata = []
        for dicom_file in dicom_file_lst:
            data = pydicom.dcmread(dicom_file)
            dcm_metadata.append(data)
        return dcm_metadata
    

    def get_first_of_dicom_field_as_int(self, x):
        """
        get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        """
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def get_windowing(self, metadata):
        """
        Get window setting metadata among the metametadata out of a DICOM image
        """
        dicom_fields = [metadata[('0028', '1050')].value,  # window center
                        metadata[('0028', '1051')].value,  # window width
                        metadata[('0028', '1052')].value,  # intercept
                        metadata[('0028', '1053')].value]  # slope

        return [self.get_first_of_dicom_field_as_int(x) for x in dicom_fields]


    def label_set(self, directory, start, size):
        """
        Obtains a set of train labels and add two columns for Sub_type and Patient_ID
        Then, sort per ID column and take labels in between start and start + size
        """
        train_labels = pd.read_csv(os.path.join(self.path, directory))
        train_labels['Sub_type'] = train_labels['ID'].str.split("_", n=3, expand=True)[
            2]
        train_labels['PatientID'] = train_labels['ID'].str.split("_", n=3, expand=True)[
            1]
        train_labels = train_labels.sort_values('ID')
        labels = train_labels[start:start+size]

        return labels


    def pixel_to_hounsefield(self, dcm, metadata):
        return self.window_image(dcm, self.get_windowing(metadata))


    @classmethod
    def read_dcm_files(cls, directory):
        """
        Read all .dcm files into train and text
        """
        train = sorted(glob(directory))
        return train


    def resizing(self, dcm_data, desired_size):
        """
        Take a DICOM metadata containing pixel_array and resize its pixel_array
        in the shape of desired_size, for example (128, 128). 
        Correspondingly, the rows and columns are adjusted.
        """
        updated_dcm_data = dcm_data
 
        try:
            img = self.three_channel_window(dcm_data)
        except:
            img = np.zeros(desired_size)
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
        updated_dcm_data.Rows = desired_size[0]
        updated_dcm_data.Columns = desired_size[1]    

        return (img, updated_dcm_data)


    def three_channel_window(self, dcm):
        # metadata = pydicom.dcmread(dcm)
        window_center, window_width, intercept, slope = self.get_windowing(dcm)
        brain_img = self.window_image(dcm, 40, 80, intercept, slope)
        subdural_img = self.window_image(dcm, 80, 200, intercept, slope)
        soft_img = self.window_image(dcm, 40, 380, intercept, slope)

        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img + 20) / 200
        soft_img = (soft_img + 150) / 380
        tcw_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

        return tcw_img


    def transform_all_pixel_arrays(self, dicom_file_lst, pixel_arr):
        """
        Return a normalized(0,1) version of the DICOM files as a list.
        """
        normalized_pixel_arrays = []
        for i in range(len(dicom_file_lst)):
            window_center, window_width, intercept, slope = self.get_windowing(dicom_file_lst[i])
            # pixel_array_HU = self.window_image(
            #     pixel_arr, window_center, window_width, intercept, slope)
            pixel_array_normalized = self.decimalize(pixel_arr, window_width)
            normalized_pixel_arrays.append(pixel_array_normalized)
        return normalized_pixel_arrays

        # for dicom_file in dicom_file_lst:
        #     data = pydicom.dcmread(dicom_file)
        #     window_center, window_width, intercept, slope = self.get_windowing(
        #         data)
        #     pixel_array = data.pixel_array
        #     pixel_array_HU = self.window_image(
        #         data, window_center, window_width, intercept, slope)
        #     pixel_array_normalized = self.decimalize(pixel_array_HU, window_width)

        #     normalized_pixel_arrays.append(pixel_array_normalized)
        


    def window_image(self, dcm, window_center, window_width, intercept, slope):
        """
        Transform the pixel values to Hounsefield units (HU)
        pixel_arr = a pixel_array for one of patient
        window_center = 36 // is the part of Metadata of the DICOM file
        window_width = 80  // is the part of Metadata of the DICOM file
        intercept = -1024  // is the part of Metadata of the DICOM file
        slope = 1          // is the part of Metadata of the DICOM file
        """

        pixel_array_HU = dcm.pixel_array.copy()

        pixel_array_HU = (pixel_array_HU*slope + intercept)
        pixel_array_HU_min = window_center - window_width//2
        pixel_array_HU_max = window_center + window_width//2

        # removing outliers -->
        # how do we deal with outliers?
        # What if the following two lines of code remove meaningful data?
        pixel_array_HU[pixel_array_HU <
                       pixel_array_HU_min] = pixel_array_HU_min
        pixel_array_HU[pixel_array_HU >
                       pixel_array_HU_max] = pixel_array_HU_max

        return pixel_array_HU