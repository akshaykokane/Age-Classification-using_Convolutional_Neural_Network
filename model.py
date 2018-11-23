from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import glob
from google.protobuf.json_format import MessageToJson
tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.as_default()

def parse_record(record):
    print(record)
    keys_to_features = {
        "image_raw" : tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    }
    try:
        parsed_record = tf.parse_single_example(record, keys_to_features)
    except ValueError:
        print("Oops!  The feature does not exists!.  Try again...")
        raise

    print(parsed_record)
    image = tf.decode_raw(parsed_record["image_raw"], tf.uint8) #reinterpret the bytes of a string as a vector
    image = tf.cast(image, tf.float32) #check if your model needs float
    image = tf.reshape(image, shape=[227,227,3])

    label = tf.cast(parsed_record["label"], tf.int32)

    return image, label

def input_fn(filenames, train=1, batch_size=32, buffer_size=2048):
   dataset = tf.data.TFRecordDataset(filenames=filenames)
   dataset = dataset.map(parse_record)
   print(dataset)
   if train:
       dataset = dataset.shuffle(buffer_size=buffer_size)
       num_repeat = None
   else:
       num_repeat = 1

   dataset = dataset.repeat(num_repeat)
   dataset = dataset.batch(batch_size)
   iterator = dataset.make_one_shot_iterator()

   images_batch, labels_batch = iterator.get_next()

   x = {'image': images_batch}
   y = labels_batch
   return x, y


def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecords"], train=True)

def val_input_fn():
    return input_fn(filenames=["val.tfrecords"], train=False)


## ------------------------------
## Function to print images start
## ------------------------------
features, labels = train_input_fn()
# Initialize `iterator` with training data.
#sess.run(train_iterator.initializer)
# Initialize `iterator` with validation data.
#sess.run(val_iterator.initializer)
img, label = sess.run([features['image'], labels])
print(img.shape, label.shape)
 # Loop over each example in batch
for i in range(img.shape[0]):
    cv2.imshow('image', img[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('Class label ' + str(np.argmax(label[i])))
## ------------------------------
## Function to print images end
## ------------------------------
print("Done!!!!!!!!!!!!!!")
