from random import shuffle
import glob
import sys
import cv2
import numpy as np
#import skimage.io as io
import tensorflow as tf

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
    #cv2.waitKey(200)
    #cv2.destroyAllWindows()
    return img


def createDataRecord(out_filename, addrs, labels):
    print(len(addrs))
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):

        # print how many images are saved every 1000 images
        if not i % 1:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

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

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


face_data = 'C:/Users/akshay1/Desktop/Books/CV Project/dataset_2/aligned/*/*.jpg'
# read addresses and labels from the 'train1' folder
addrs = glob.glob(face_data)
labels = [0 if '20' in addr else 1 for addr in addrs]

#read labels from text file
labels_file_fold0 = open("dataset_2/fold_0_data.txt", "r")
labels_file_fold1 = open("dataset_2/fold_1_data.txt", "r")
labels_file_fold2 = open("dataset_2/fold_2_data.txt", "r")
labels_file_fold3 = open("dataset_2/fold_3_data.txt", "r")
labels_file_fold4 = open("dataset_2/fold_4_data.txt", "r")

print(labels_file_fold4.read(0))

#for i in range(len(addrs)):
#    print(addrs[i])

# to shuffle data
#c = list(zip(addrs, labels))
#shuffle(c)
#addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]
val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]

#createDataRecord('train.tfrecords', train_addrs, train_labels)
#createDataRecord('val.tfrecords', val_addrs, val_labels)
#createDataRecord('test.tfrecords', test_addrs, test_labels)
