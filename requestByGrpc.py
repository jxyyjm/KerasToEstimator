#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
  ===> grpc request 
  request the predict serving and get the probability
  ===> model is saved by estimator.export_saved_model
  serving_input_receiver_fn has been defined 
  here data is tf.train.Example
  reference : https://stackoverflow.com/questions/53888284/how-to-send-a-tf-example-into-a-tensorflow-serving-grpc-predict-request
  notice: TF-Version 1.15.0
'''
import sys 
import grpc
import math
import time
import numpy as np
import request_pb2
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer('concurrency', 1, 'max num of concur inf req')
tf.app.flags.DEFINE_integer('num_tests', 10000, 'num of test-case')
tf.app.flags.DEFINE_string('server', '', 'PredServing host:port')
tf.app.flags.DEFINE_string('work_dir', './tmp/', 'work directory')
FLAGS = tf.app.flags.FLAGS
#seq_len = 14
#item_len = 4 
#user_len = 1 

def do_inference(hostport, work_dir, concurrency, num_tests):
  tf_ex = tf.train.Example(
    features=tf.train.Features(
        feature={
            'feat1': tf.train.Feature(float_list=tf.train.FloatList(value=[0.1])),
            'feat2': tf.train.Feature(float_list=tf.train.FloatList(value=[0.2])),
            'feat3': tf.train.Feature(float_list=tf.train.FloatList(value=[0.3])),
            'feat4': tf.train.Feature(float_list=tf.train.FloatList(value=[0.4])),
        }   
    )   
  )
  channel = grpc.insecure_channel('0.0.0.0:8500')
  stub    = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'model' ## model name
  request.model_spec.signature_name = 'serving_default' ## signature_def_map ##
  #request.inputs['inputs'].CopyFrom(tf.make_tensor_proto([tf_ex.SerializeToString()])) ## for original tf.estimator.export_save_model ##
  request.inputs['examples'].CopyFrom(tf.make_tensor_proto([tf_ex.SerializeToString()])) ## for tf.keras.Model translate tf.estimator ##
  response = stub.Predict(request, 5.0)
  print ('response', response)
  print ('type', type(response))
  print ('response.outputs', response.outputs)
  if 'scores' in response.outputs:
    res1 = tf.make_ndarray(response.outputs['scores'])
    print ('score-res', res1)
  if 'classes' in response.outputs:
    res2 = tf.make_ndarray(response.outputs['classes'])
    print ('class-res', res2)
def main(_):
  if FLAGS.num_tests > 10001:
    print ('num_test should not be gt 10k')
    return
  #if not FLAGS.server:
  #  print ('please specify server host:port')
  #  return
  do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)

if __name__=='__main__':
  tf.app.run()
