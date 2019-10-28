import pandas as pd
import numpy as np
from glob import glob
import os
import re
from PIL import Image
from random import randrange
import pydicom
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_csv(file_name_csv):
    labels = get_labels(file_name_csv)
    print(f'csv file loaded')
    return labels


def load_data(batch_index, BATCH_SIZE, all_labels, train, quite=True):
    # file_name_csv = 'first_6000_training_labels.csv'
    # labels = get_labels(file_name_csv)[batch_index*BATCH_SIZE : 6*BATCH_SIZE*(batch_index+1)]

    labels = all_labels[6*BATCH_SIZE*batch_index : 6*BATCH_SIZE*(batch_index+1)]
    label_column_np = np.array(labels['Label'])
    if not quite: print(f'Label column extracted and converted to NumPy array')
    category_label_list = get_category_numbers(label_column_np)
    if not quite: print(f'Label column size reduce from 6,000 to 1,000')

    train = train[batch_index*BATCH_SIZE : BATCH_SIZE*(batch_index+1)]
    if not quite: print(f'train batch image filenames loaded')
    datas  = get_datas(train)

    if not quite: print(f'datas loaded')
    normalized_pixel_arrays = transform_all_pixel_arrays_2(datas)
    if not quite: print(f'normalized_pixel_arrays')

    training_images = np.array(normalized_pixel_arrays)
    training_labels = np.array(category_label_list)

    training_images = training_images.reshape(*training_images.shape, 1)
    if not quite: print(f'reshaping training_images: {training_images.shape}')
    if not quite: print(f'training_labels: {training_labels.shape}')

    return training_images, training_labels


def get_labels(file_name):
    """
    Obtains a set of train labels and add two columns for Sub_type and Patient_ID
    Then, sort per ID column and take the first 6,000 labels
    """
    train_labels = pd.read_csv(file_name)
    train_labels['Sub_type']  = train_labels['ID'].str.split("_", n=3, expand=True)[2]
    train_labels['PatientID'] = train_labels['ID'].str.split("_", n=3, expand=True)[1]
    train_labels = train_labels.sort_values('ID')
    # labels = train_labels[:6000]
    labels = train_labels
    
    return labels


def read_dcm_files(directory):
    """
    Read all .dcm files into train and text
    """
    train = sorted(glob(directory))
    return train

# def read_dcm_image(self, dcm_file_name):
#     """Read a DICOM header(Metadata) and pixel_array from a file
#     """
#     data = pydicom.dcmread(dicom_file)

#     return data

def resize_dicom_images(datas, new_img_px_size):
    NEW_IMG_PX_SIZE = new_img_px_size
    for data in datas:
        if data.pixel_array.shape != (new_img_px_size, new_img_px_size):
            data.pixel_array = resize(data.pixel_array, (NEW_IMG_PX_SIZE, NEW_IMG_PX_SIZE), anti_aliasing=True)
        else:
            return dicom_image  

def get_datas(dicom_file_lst):
    """
    Return a list of DICOM headers(Metadatas) and pixel_arryas from the DCM file list
    """
    datas = []
    for dicom_file in dicom_file_lst:
        data = pydicom.dcmread(dicom_file)
        datas.append(data)
    return datas


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
    

def display_img_HU(pixel_img, window_center, window_width, intercept, slope):
    """
    displays a image transformed into Hounsefield Units from pixel values
    """
    HU_img = window_image(pixel_img, window_center, window_width, intercept, slope)
    plt.imshow(HU_img, cmap=plt.cm.bone)
    plt.grid(False)


def get_first_of_dicom_field_as_int(dicom_field):
    """
    get dicom_field[0] as int if dicom_field is a 'pydicom.multival.MultiValue', otherwise get int(dicom_field)
    """
    if type(dicom_field) == pydicom.multival.MultiValue:
        return int(dicom_field[0])
    else:
        return int(dicom_field)


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
    array_HU_shifted = image_HU - min # array_HU is shifted by min

    array_HU_decimalized = array_HU_shifted / (max - min)
    return array_HU_decimalized


def transform_all_pixel_arrays(dicom_file_lst):
    '''Return a normalized(0,1) version of the DICOM files as a list.
    '''
    normalized_pixel_arrays = []
    for dicom_file in dicom_file_lst:
        data = pydicom.dcmread(dicom_file)
        window_center, window_width, intercept, slope = get_windowing(data)
        pixel_array = data.pixel_array
        pixel_array_HU = window_image(pixel_array, window_center, window_width, intercept, slope)
        min = window_center - window_width // 2
        pixel_array_HU_shifted = pixel_array_HU - min
        pixel_array_normalized = pixel_array_HU_shifted / window_width
        
        normalized_pixel_arrays.append(pixel_array_normalized)
    return normalized_pixel_arrays


def transform_all_pixel_arrays_2(datas):
    '''Return a normalized(0,1) version of the DICOM files as a list.
    '''
    normalized_pixel_arrays = []
    for data in datas:
        window_center, window_width, intercept, slope = get_windowing(data)
        pixel_array = data.pixel_array
        pixel_array_HU = window_image(pixel_array, window_center, window_width, intercept, slope)
        min = window_center - window_width // 2
        pixel_array_HU_shifted = pixel_array_HU - min
        pixel_array_normalized = pixel_array_HU_shifted / window_width
        
        normalized_pixel_arrays.append(pixel_array_normalized)
    return normalized_pixel_arrays

def get_category_number(sub_types):
    i, total = 0, 0
    for t in sub_types:
        # print(f'type(t): {type(t)}')
        total = total + 2**i * t
        i = i + 1
    return total


def get_category_numbers(labels):
    category_label_lst = []

    for j in range(0, len(labels), 6):
        sub_types = labels[j: j+6]
        category_number = get_category_number(sub_types)
        category_label_lst.append(category_number)
    return category_label_lst

to_subtypes = lambda N: [int(ch) for ch in bin(N)[2:].ljust(6, '0')]

def reduce_labels(labels):
    category_label_lst = []

    for i in range(0, len(labels), 6):
        lst = labels[i: i+6]
        found = False
        index = -1
        for ind in range(6):
            if lst[ind] == 1:
                found = True
                index = ind
        if found == True:
            category_label_lst.append(index)
        else:
            category_label_lst.append(6)
    return category_label_lst
