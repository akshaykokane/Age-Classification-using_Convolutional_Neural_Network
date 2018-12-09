from random import shuffle
import glob
import sys
import cv2
import numpy as np

import tensorflow as tf

#age groups having labels to value mapping
age_groups = {"1":"(0,2)", "2":"(4,6)", "3":"(8,13)","4":"(15,20)","5":"(25,32)","6":"(38,43)","7":"(48,53)","8":"(60,75)"}

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img=cv2.imread(addr)
    if img is None:
        return None
    #resize the image to 256x256
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    #crop the image 227x227
    img = img[29:256, 29:256]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Check if images are getting read properly
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', img)
    #cv2.waitKey(300)
    #cv2.destroyAllWindows()
    return img


def createDataRecord(out_filename, addrs, labels):
    print(len(addrs))
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):

        img = load_image(addrs[i])

        print(labels[i])
        label = labels[i]

        if img is None:
            break

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
       
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        if not i % 1000:
         print(example)
       
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
   


face_data = 'C:/Users/akshay1/Desktop/Books/CV Project/dataset_2/aligned/*/*.jpg'
#face_data = 'C:/Users/akshay1/Desktop/Books/CV Project/landmark_aligned_face.1938.8496685134_bafd8bda68_o.jpg'

# read addresses and labels from the 'train1' folder
addrs = glob.glob(face_data)
print(len(addrs))

#read labels from text file
labels_file_fold0 = open("dataset_2/extracted_age_dataset/fold_0_age_labels.txt", "r")
labels_file_fold1 = open("dataset_2/extracted_age_dataset/fold_1_age_labels.txt", "r")
labels_file_fold2 = open("dataset_2/extracted_age_dataset/fold_2_age_labels.txt", "r")
labels_file_fold3 = open("dataset_2/extracted_age_dataset/fold_3_age_labels.txt", "r")
labels_file_fold4 = open("dataset_2/extracted_age_dataset/fold_4_age_labels.txt", "r")

file_names = []
file_labels = []
i=0
list_of_files=[labels_file_fold0,labels_file_fold1,labels_file_fold2,labels_file_fold3,labels_file_fold4]

for current_file in list_of_files:
    for words in current_file.read().split():
       if i%2==0:
           file_names.insert((i-1),words)
       else:
           file_labels.insert(i,words)
       i=i+1

#print(len(file_labels))
#print(len(file_names))

labels = []
for addr in addrs:
    flag = 0
    for i in range(0,len(file_names)):
        if file_names[i] in addr:
            break
    print(i)
    if file_labels[i]!=-1:
        labels.append(int(file_labels[i]))
    else:
        labels.append(int(0))



#print(labels)




#for i in range(len(addrs)):
#    print(addrs[i])

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 40% test
#train_addrs = addrs[0:int(0.6 * len(addrs))]
#train_labels = labels[0:int(0.6 * len(labels))]
#val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
#val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
#test_addrs = addrs[int(0.8 * len(addrs)):]
#test_labels = labels[int(0.8 * len(labels)):]

#createDataRecord('train.tfrecords', train_addrs, train_labels)
#createDataRecord('val.tfrecords', val_addrs, val_labels)
#createDataRecord('test.tfrecords', test_addrs, test_labels)
createDataRecord('visual.tfrecords', addrs, labels)
