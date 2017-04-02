from __future__ import print_function

import numpy as np
import tflearn
import dicom
import os


# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)


# Preprocessing function
def preprocess_many(data):
    # new array
    data_new = []
    # Load dicom's from folder
    for item in data:
        temp_item = []
        
        # # Get 1st file
        # RefDs = dicom.read_file(item[1])
        # # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        # ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
        # # Load spacing values (in mm)
        # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        # ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        # # read the file
        ds = dicom.read_file(item[1])

        # Get 1st file
        # RefDs = dicom.read_file(item[2])
        # # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        # ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
        # # Load spacing values (in mm)
        # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
        # ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        # # read the file
        ds2 = dicom.read_file(item[2])
        
        temp_item.append(item[0])
        temp_item.append(ds.pixel_array)
        temp_item.append(ds2.pixel_array)
        print(ds.pixel_array)
        data_new.append(temp_item)
    return np.array(data_new, dtype=np.float32)

# Preprocessing function
def preprocess_one(data):
    # Load dicom's from folder
    for item in data:
        print(item)
    return np.array(data, dtype=np.float32)


# TODO: to external csv(?) file this:
## Preprocess data
## first parameter is true or false
## 2nd - profile dicom
## 3d - front dicom
data = preprocess_many([
    [1,'data/N5836DM.di','data/N5836DM-1.di'], 
    [0,'data/N5806DF.di','data/N5806DF-1.di']
])

# Build neural network
# Here 3 is columns in array
net = tflearn.input_data(shape=[None, 3])
# 1st layer init
net = tflearn.fully_connected(net, 32)
# 2nd layer init
net = tflearn.fully_connected(net, 32)
# 3rd layer init
net = tflearn.fully_connected(net, 2, activation='softmax')
# regresion net
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# data = preprocess_one(['data/filename.dicom','data/filename.dicom'])
# pred = model.predict(data)
# print(pred)