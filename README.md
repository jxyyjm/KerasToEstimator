### KerasToEstimator
1) tf-2x.keras is good enough for beginer

   here give an example translate tf.keras.Model into tf.estimator
   
2) what is different

   keras.Model input has been point by input-dict using tf.keras.layers.DenseFeatures(feature_columns)(keras.dict_input).
   
   both could define the signature_def_map method saving model to receive the request data. [in this repostory "seqfirstpos"](https://github.com/jxyyjm/seqfirstpos)
   
   both could define the serving_input_receiver_fn method saving model to receive the request data. [here give the code in KerasSaveModel.py]
   
### request 
1) grpc request using pb
2) http request 
