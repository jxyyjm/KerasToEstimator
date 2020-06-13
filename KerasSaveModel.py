#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
  1) use tf.keras.Model to build-model, then transInto tf.estimator to save 
  2) serving_input_receiver_fn is defined here and receive the pb data
  reference: https://github.com/tensorflow/tensorflow/issues/28976
  Notice: TF-Verion 2.0.0
'''
import os
import time
import numpy as np
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

def build_column():
  feat1_column = tf.feature_column.numeric_column('feat1')
  feat2_column = tf.feature_column.numeric_column('feat2')
  feat3_column = tf.feature_column.numeric_column('feat3')
  feat4_column = tf.feature_column.numeric_column('feat4')
  feature_columns = [feat1_column, feat2_column, feat3_column, feat4_column]
  return feature_columns

#def input_fn(x, y):
#  dataset = tf.data.Dataset.from_tensor_slices((x, y))
#  dataset = dataset.shuffle(buffer_size=100).batch(16)
#  return dataset

def input_fn(file_name):
  def decode_line(line):
    columns_value = tf.io.decode_csv(line, record_defaults=[0.0, 0.0, 0.0, 0.0, 0], field_delim='\t')
    return dict(zip(['feat1', 'feat2', 'feat3', 'feat4'], columns_value[0:4])), columns_value[-1]
    #return dict(zip(['feat1', 'feat2', 'feat3', 'feat4', 'label'], columns_value))
  dataset = tf.data.TextLineDataset(file_name)
  dataset = dataset.map(decode_line)
  dataset = dataset.batch(16)
  return dataset

def _serving_input_receiver_fn():
  input_feature = { \ 
    'feat1': tf.io.FixedLenFeature([1], tf.float32), \
    'feat2': tf.io.FixedLenFeature([1], tf.float32), \
    'feat3': tf.io.FixedLenFeature([1], tf.float32), \
    'feat4': tf.io.FixedLenFeature([1], tf.float32), \
  }
  tf.compat.v1.disable_eager_execution()
  serialized_tf_example = tf.compat.v1.placeholder( \
                  dtype = tf.string, \
                  shape = (None), \
                  name  = 'input_example_tensor')
  features = tf.io.parse_example(serialized_tf_example, input_feature)
  receiver_tensors = {'examples': serialized_tf_example}
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
  #return tf.estimator.export.TensorServingInputReceiver(features, receiver_tensors)

def main():
  ## Keras Model create ##
  input1 = tf.keras.Input(shape=(1,), name='feat1', dtype=tf.float32)
  input2 = tf.keras.Input(shape=(1,), name='feat2', dtype=tf.float32)
  input3 = tf.keras.Input(shape=(1,), name='feat3', dtype=tf.float32)
  input4 = tf.keras.Input(shape=(1,), name='feat4', dtype=tf.float32)
  inputs = {'feat1':input1, 'feat2':input2, 'feat3':input3, 'feat4':input4}
  ## keras could not map the input by your feature-names
  ## so here need to point the dict of names and input 
  
  layer  = tf.keras.layers.DenseFeatures(build_column())(inputs)
  output = tf.keras.layers.Dense(1, name='scores')(layer)
  KModel = tf.keras.Model(inputs=inputs, outputs=output)
  KModel.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
  estimator = tf.keras.estimator.model_to_estimator(keras_model = KModel, model_dir='../model/', config=None)
  for i in range(3):
    estimator.train(input_fn=lambda: input_fn('./random.data'))#, steps=500)
    accu = estimator.evaluate(input_fn=lambda: input_fn('./random.data.test'), steps=100)
    print ('i=', i, 'accu', accu)

  tf.compat.v1.disable_eager_execution()
  estimator.export_saved_model( \
    export_dir_base = '../model/lastest/', \
    serving_input_receiver_fn = _serving_input_receiver_fn
  )
  
  #tf.compat.v1.enable_eager_execution()

  params = {}
  #params['columns'] = build_column()
  #data_train = input_fn('./random.data', 2, True, 16)

if __name__ =='__main__':
  gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  #tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
  main()
