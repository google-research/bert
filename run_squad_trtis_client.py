# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import modeling
import tokenization
from tensorrtserver.api import ProtocolType, InferContext, ServerStatusContext, grpc_service_pb2_grpc, grpc_service_pb2, model_config_pb2
from utils.create_squad_data import *
import grpc
from run_squad import *
import numpy as np
import tqdm

# Set this to either 'label_ids' for Google bert or 'unique_ids' for JoC
label_id_key = "unique_ids"

PendingResult = collections.namedtuple("PendingResult",
                                   ["async_id", "start_time", "inputs"])

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        label_ids_data = ()
        input_ids_data = ()
        input_mask_data = ()
        segment_ids_data = ()
        for i in range(0, min(n, l-ndx)):
            label_ids_data = label_ids_data + (np.array([iterable[ndx + i].unique_id], dtype=np.int32),)
            input_ids_data = input_ids_data+ (np.array(iterable[ndx + i].input_ids, dtype=np.int32),)
            input_mask_data = input_mask_data+ (np.array(iterable[ndx + i].input_mask, dtype=np.int32),)
            segment_ids_data = segment_ids_data+ (np.array(iterable[ndx + i].segment_ids, dtype=np.int32),)

        inputs_dict = {label_id_key: label_ids_data,
                       'input_ids': input_ids_data,
                       'input_mask': input_mask_data,
                       'segment_ids': segment_ids_data}
        yield inputs_dict

def run_client():
    """
    Ask a question of context on TRTIS.
    :param context: str
    :param question: str
    :param question_id: int
    :return:
    """

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False,
        version_2_with_negative=FLAGS.version_2_with_negative)

    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples[0:],
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)

    protocol_str = 'grpc' # http or grpc
    url = FLAGS.trtis_server_url
    verbose = True
    model_name = FLAGS.trtis_model_name
    model_version = FLAGS.trtis_model_version
    batch_size = FLAGS.predict_batch_size

    protocol = ProtocolType.from_str(protocol_str) # or 'grpc'

    ctx = InferContext(url, protocol, model_name, model_version, verbose)

    channel = grpc.insecure_channel(url)

    stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

    prof_request = grpc_service_pb2.server__status__pb2.model__config__pb2.ModelConfig()

    prof_response = stub.Profile(prof_request)

    status_ctx = ServerStatusContext(url, protocol, model_name=model_name, verbose=verbose)

    model_config_pb2.ModelConfig()

    status_result = status_ctx.get_server_status()

    outstanding = {}
    max_outstanding = 20

    sent_prog = tqdm.tqdm(desc="Send Requests", total=len(eval_features))
    recv_prog = tqdm.tqdm(desc="Recv Requests", total=len(eval_features))

    def process_outstanding(do_wait):

        if (len(outstanding) == 0):
            return
        
        ready_id = ctx.get_ready_async_request(do_wait)

        if (ready_id is None):
            return

        # If we are here, we got an id
        result = ctx.get_async_run_results(ready_id, False)
        stop = time.time()

        if (result is None):
            raise ValueError("Context returned null for async id marked as done")

        outResult = outstanding.pop(ready_id)

        time_list.append(stop - outResult.start_time)

        batch_count = len(outResult.inputs[label_id_key])

        for i in range(batch_count):
            unique_id = int(outResult.inputs[label_id_key][i][0])
            start_logits = [float(x) for x in result["start_logits"][i].flat]
            end_logits = [float(x) for x in result["end_logits"][i].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

        recv_prog.update(n=batch_count)

    all_results = []
    time_list = []

    print("Starting Sending Requests....\n")

    all_results_start = time.time()

    for inputs_dict in batch(eval_features, batch_size):

        present_batch_size = len(inputs_dict[label_id_key])

        outputs_dict = {'start_logits': InferContext.ResultFormat.RAW,
                        'end_logits': InferContext.ResultFormat.RAW}

        start = time.time()
        async_id = ctx.async_run(inputs_dict, outputs_dict, batch_size=present_batch_size)

        outstanding[async_id] = PendingResult(async_id=async_id, start_time=start, inputs=inputs_dict)

        sent_prog.update(n=present_batch_size)

        # Try to process at least one response per request
        process_outstanding(len(outstanding) >= max_outstanding)

    tqdm.tqdm.write("All Requests Sent! Waiting for responses. Outstanding: {}.\n".format(len(outstanding)))

    # Now process all outstanding requests
    while (len(outstanding) > 0):
        process_outstanding(True)

    all_results_end = time.time()
    all_results_total = (all_results_end - all_results_start) * 1000.0

    print("-----------------------------")
    print("Individual Time Runs - Ignoring first two iterations")
    print("Total Time: {} ms".format(all_results_total))
    print("-----------------------------")

    print("-----------------------------")
    print("Total Inference Time = %0.2f for"
          "Sentences processed = %d" % (sum(time_list), len(eval_features)))
    print("Throughput Average (sentences/sec) = %0.2f" % (len(eval_features) / all_results_total * 1000.0))
    print("-----------------------------")

    time_list.sort()

    avg = np.mean(time_list)
    cf_95 = max(time_list[:int(len(time_list) * 0.95)])
    cf_99 = max(time_list[:int(len(time_list) * 0.99)])
    cf_100 = max(time_list[:int(len(time_list) * 1)])
    print("-----------------------------")
    print("Summary Statistics")
    print("Batch size =", FLAGS.predict_batch_size)
    print("Sequence Length =", FLAGS.max_seq_length)
    print("Latency Confidence Level 95 (ms) =", cf_95 * 1000)
    print("Latency Confidence Level 99 (ms)  =", cf_99 * 1000)
    print("Latency Confidence Level 100 (ms)  =", cf_100 * 1000)
    print("Latency Average (ms)  =", avg * 1000)
    print("-----------------------------")


    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)



if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  run_client()

