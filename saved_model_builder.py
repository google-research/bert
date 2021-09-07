import os
# Execute in Tf 1.15
import tensorflow.compat.v1 as tf
#import tensorflow as tf
import numpy as np
import time
from tensorflow.core.protobuf import rewriter_config_pb2

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 8,
                     "Total batch size for predictions.")
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("trt", False, "Whether to use TensorRT.")

flags.DEFINE_bool("amp", False, "Mixed Precision for Float 16.")

flags.DEFINE_bool("xla", False, "Acclerated Linear Algebra")

# From Nvidia Repo, explained here: https://github.com/NVIDIA/DeepLearningExamples/issues/57
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
# Enable AMP
amp = FLAGS.amp
xla = FLAGS.xla
if amp:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
if xla:
    # https://github.com/tensorflow/tensorflow/blob/8d72537c6abf5a44103b57b9c2e22c14f5f49698/tensorflow/compiler/jit/flags.cc#L78-L87
    # 1: on for things very likely to be improved
    # 2: on for everything
    # fusible: only for Tensorflow operations that XLA knows how to fuse
    #
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    # Best Performing XLA Option
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'
    os.environ["TF_XLA_FLAGS"] = (os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_enable_lazy_compilation=false")

tf.compat.v1.disable_eager_execution()
model_dir = '/home/ubuntu/bert/saved_model/1630972186'
trt_model_dir = '/home/ubuntu/trt_model/'
batch_size = FLAGS.batch_size
seq_length = FLAGS.max_seq_length
use_trt = FLAGS.trt

warm_up = 50
total_iterations = 1500

fake_data = {
      "Placeholder:0": np.ones((batch_size,)),
      "Placeholder_1:0": np.ones((batch_size, seq_length)),
      "Placeholder_2:0": np.ones((batch_size, seq_length)),
      "Placeholder_3:0": np.ones((batch_size, seq_length))
}

def input_map_fn():
    return {
          "Placeholder:0":  tf.convert_to_tensor(np.ones((batch_size,)), dtype=tf.int32),
          "Placeholder_1:0":  tf.convert_to_tensor(np.ones((batch_size, seq_length)), dtype=tf.int32),
          "Placeholder_2:0":  tf.convert_to_tensor(np.ones((batch_size, seq_length)), dtype=tf.int32),
          "Placeholder_3:0":  tf.convert_to_tensor(np.ones((batch_size, seq_length)), dtype=tf.int32)}


session_config = tf.GraphOptions(
    rewrite_options=rewriter_config_pb2.RewriterConfig(
      auto_mixed_precision=1 if FLAGS.amp else 2,
  ))

session_config = tf.ConfigProto(
          graph_options=session_config,
          allow_soft_placement=True)

time_list = []
steps = 0


with tf.Session(config=session_config) as sess:
    if use_trt:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        converter = trt.TrtGraphConverter(
            input_saved_model_dir=model_dir,
            max_workspace_size_bytes=(11<32),
            precision_mode='INT8',
            maximum_cached_engines=100,
            max_batch_size=batch_size,
            is_dynamic_op=True,
            use_calibration=True)

        frozen_graph = converter.convert()
        frozen_graph = converter.calibrate(
        fetch_names=['unstack:0', 'unstack:1'],
        num_runs=100,
        input_map_fn=input_map_fn)
        converter.save(trt_model_dir)

        output_node = tf.import_graph_def(
        frozen_graph,
        return_elements=['unstack:0', 'unstack:1'])
        key_list = list(fake_data.keys())
        for key in key_list:
            value = fake_data[key]
            del fake_data[key]
            fake_data["import/" + key] = value
        num_converted_nodes = len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp'])
        print("Converted Nodes: " + str(num_converted_nodes))
    else:
        # First load the SavedModel into the session    
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING],
            model_dir)
        # Output Tensors from Bert Graph (start_logits, end_logits) for Prediction
        output_0 = tf.get_default_graph().get_tensor_by_name("unstack:0")
        output_1 = tf.get_default_graph().get_tensor_by_name("unstack:1")
        output_node = [output_0, output_1]

    for i in range(total_iterations):
        if i < warm_up:
            output = sess.run(output_node, feed_dict=fake_data)
            continue
        else:
            temp = time.time()
            output = sess.run(output_node, feed_dict=fake_data)
            time_list.append(time.time()-temp)
            steps += batch_size
        if i%100 == 0:
            print("Iteration " + str(i))

duration_ms = np.array(time_list)
mean_latency = np.mean(duration_ms)
p99_latency = np.quantile(duration_ms, 0.99)
p95_latency = np.quantile(duration_ms, 0.95)
p90_latency = np.quantile(duration_ms, 0.90)
throughput = steps / float(sum(time_list))
print('Examples: {:0.5f}'.format(float(steps)))
print('Time Passed: {:0.5f}s'.format(sum(time_list)))
print('Throughput: {:0.5f} eps'.format(throughput))
print('Mean Latency: {:0.5f}s'.format(mean_latency))
print('P90 Latency: {:0.5f}s'.format(p90_latency))
print('P95 Latency: {:0.5f}s'.format(p95_latency))
print('P99 Latency: {:0.5f}s'.format(p99_latency))