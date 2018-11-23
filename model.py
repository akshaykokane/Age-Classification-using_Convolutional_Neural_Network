from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import glob
import sys
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


def cnn_model_fn(features, labels, mode, params):
    input_layer = tf.reshape(features["image"], [-1, 227, 227, 3])
    input_layer = tf.identity(input_layer, name="input_tensor")
    #Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2)

    pool3_flat=tf.reshape(pool2, [-1, 14 * 14 * 256])

    # Dense Layer - 1
    dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

    # DropOut - 1
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)



    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout1, units=8)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Create the Estimator
age_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/age_convnet_model")


# Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

count = 0
while (count < 1):
    age_classifier.train(input_fn=train_input_fn, steps=1000)
    result = age_classifier.evaluate(input_fn=val_input_fn)
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    sys.stdout.flush()
    count = count + 1