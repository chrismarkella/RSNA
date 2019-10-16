import pandas as pd
import numpy as np
from glob import glob
import os
import re
from PIL import Image
import seaborn as sns
from random import randrange
import pydicom
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, path):
        self.path = path
    

    def label_set (self, directory):
        """
        Obtains a set of train labels and add two colums for Sub_type and Patient_ID
        Then, sort per 'ID' column and take the first 6,000 labels
        """
        train_labels = pd.read_csv(os.path.join(self.path, directory))
        train_labels['Sub_type'] = train_labels['ID'].str.split("_", n=3, expand=True)[2]
        train_labels['PatientID'] = train_labels['ID'].str.split("_", n=3, expand=True)[1]
        train_labels = train_labels.sort_values('ID')
        labels = train_labels[:6000]
        
        return labels


    def read_dcm_files(self, directory):
        """
        Read all .dcm files into train and text
        """
        train = sorted(glob(join(self.path, directory)))
        return train


    def display_img_HU(pixel_img, window_center, window_width, intercept, slope):
        """
        displays a image transformed into Hounsefield Units from pixel values
        """
        HU_img = window_image(pixel_img, window_center, window_width, intercept, slope)
        plt.imshow(HU_img, cmap=plt.cm.bone)
        plt.grid(False)


    def window_image(pixel_array, window_center, window_width, intercept, slope):
        """
        Transform the pixel values to Hounsefield units (HU)
        pixel_array = one DICOM image converted into 2-D pixel array
        window_center = 36 // is the part of Metadata of the DICOM file
        window_width = 80  // is the part of Metadata of the DICOM file
        intercept = -1024  // is the part of Metadata of the DICOM file
        slope = 1          // is the part of Metadata of the DICOM file
        """
        pixel_array_HU = pixel_array.copy()

        pixel_array_HU = (pixel_array_HU*slope + intercept)
        pixel_array_HU_min = window_center - window_width//2
        pixel_array_HU_max = window_center + window_width//2
        
        # removing outliers --> 
        # how do we deal with outliers? 
        # What if the following two lines of code remove meaningful data?
        pixel_array_HU[pixel_array_HU<pixel_array_HU_min] = pixel_array_HU_min
        pixel_array_HU[pixel_array_HU>pixel_array_HU_max] = pixel_array_HU_max
            
        return pixel_array_HU
    

    def get_first_of_dicom_field_as_int(x):
        """
        get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        """
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)


    def get_windowing(metadata):
        """
        Get window setting metadata among the metametadata out of a DICOM image
        """
        dicom_fields = [metadata[('0028','1050')].value, #window center
                        metadata[('0028','1051')].value, #window width
                        metadata[('0028','1052')].value, #intercept
                        metadata[('0028','1053')].value] #slope
                        
        return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    

    def pixel_to_hounsefield(pixel_array, metadata):
        return window_image(pixel_array, get_windowing(metadata))

    def decimalize(image_HU):
        """
        Convert values of Housefield Units into a range of (0, 1)
        image_HU: an image in Hounsefield Unit that outliers removed
        """
        min = np.amin(image_HU)
        max = np.amax(image_HU)
        array_HU_shifted = image_HU - min