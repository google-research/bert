
import grpc
import tensorflow as tf

import run_classifier as classifiers
import tokenization

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from flask import Flask
from flask import request

import random

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def hello():
  return "Hello BERT predicting AG NEWS! Try posting a string to this url"


@app.route("/", methods = ['POST'])
def predict():
  channel = grpc.insecure_channel("bert-agnews:8500")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # Parse Description
  tokenizer = tokenization.FullTokenizer(
    vocab_file="asset/vocab.txt", do_lower_case=True)
  processor = classifiers.AgnewsProcessor()
  label_list = processor.get_labels()
  content = request.get_json()
  request_id = str(random.randint(1, 9223372036854775807))
  inputExample = processor._create_example([request_id, content['description']], 'test')
  tf_example = classifiers.from_record_to_tf_example(3, inputExample, label_list, 64, tokenizer)
  model_input = tf_example.SerializeToString()

  # Send request
  # See prediction_service.proto for gRPC request/response details.
  model_request = predict_pb2.PredictRequest()
  model_request.model_spec.name = 'bert'
  model_request.model_spec.signature_name = 'serving_default'
  dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
  tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
  tensor_proto = tensor_pb2.TensorProto(
    dtype=types_pb2.DT_STRING,
    tensor_shape=tensor_shape_proto,
    string_val=[model_input])

  model_request.inputs['examples'].CopyFrom(tensor_proto)
  result = stub.Predict(model_request, 10.0)  # 10 secs timeout
  result = tf.make_ndarray(result.outputs["output"])
  pretty_result = "Predicted Label: " + label_list[result[0].argmax(axis=0)]
  app.logger.info("Predicted Label: %s", label_list[result[0].argmax(axis=0)])
  return pretty_result


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
