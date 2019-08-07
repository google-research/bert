from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
import run_squad
import json, os
import tokenization
import collections

########## Global Variable #########

app = Flask(__name__)
api = Api(app)

export_dir = <Serving Model Path> #'tf-severing_v3/1/1565064269/'
max_seq_length = 384
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128
VOCAB_FILE = <Model_Vocab_File_Path>#'../Model/vocab.txt'
DO_LOWER_CASE = True
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
answer_model = None
############ Model #####################

class Model(object):

    def __init__(self, model_path):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=tf.Graph())
        tf.saved_model.loader.load(self.session, [tag_constants.SERVING], model_path)   

    def predict(self, eval_features):
        all_results = []
        tensor_input_ids = self.session.graph.get_tensor_by_name('input_ids_1:0')
        tensor_input_mask = self.session.graph.get_tensor_by_name('input_mask_1:0')
        tensor_label_ids = self.session.graph.get_tensor_by_name('unique_ids_1:0')
        tensor_segment_ids = self.session.graph.get_tensor_by_name('segment_ids_1:0')
        tensor_outputs_end_logits = self.session.graph.get_tensor_by_name('unstack:1')
        tensor_outputs_start_logits = self.session.graph.get_tensor_by_name('unstack:0')
        tensor_outputs_start_unique_ids = self.session.graph.get_tensor_by_name('unique_ids_1:0')
        for eval_feature in eval_features:
            input_ids = eval_feature.input_ids
            input_mask = eval_feature.input_mask
            label_ids = eval_feature.unique_id
            segment_ids = eval_feature.segment_ids
            result = self.session.run([tensor_outputs_end_logits,tensor_outputs_start_logits,tensor_outputs_start_unique_ids], feed_dict={
                tensor_input_ids: np.array(input_ids).reshape(-1, max_seq_length),
                tensor_input_mask: np.array(input_mask).reshape(-1, max_seq_length),
                tensor_label_ids: np.array([label_ids]),
                tensor_segment_ids: np.array(segment_ids).reshape(-1, max_seq_length),
            })
            all_results.append(
                  run_squad.RawResult(
                  unique_id=result[2][0],
                  start_logits=result[1].tolist()[0],
                  end_logits=result[0].tolist()[0]))
        return all_results

############# API Internal Function ##############

def get_predicted_answer(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = run_squad._get_best_indexes(result.start_logits, n_best_size)
            end_indexes = run_squad._get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                      _PrelimPrediction(
                          feature_index=feature_index,
                          start_index=start_index,
                          end_index=end_index,
                          start_logit=result.start_logits[start_index],
                          end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = run_squad.get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                  _NbestPrediction(
                      text=final_text,
                      start_logit=pred.start_logit,
                      end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
              _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = run_squad._compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        all_predictions[example.qas_id] = nbest_json[0]["text"]
        '''
        # predict "" iff the null score - the score of best non-null > threshold
        score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
        scores_diff_json[example.qas_id] = score_diff
        if score_diff > 0.0:
            all_predictions[example.qas_id] = ""
        else:
            all_predictions[example.qas_id] = best_non_null_entry.text
        '''

        all_nbest_json[example.qas_id] = nbest_json
        #print(all_nbest_json)
    
    #with tf.gfile.GFile(output_prediction_file, "w") as writer:
    #    writer.write(json.dumps(all_predictions, indent=4) + "\n")

    #with tf.gfile.GFile(output_nbest_file, "w") as writer:
    #    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        
    return all_predictions

def get_squad_examples(input_data, is_training):
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  input_data = input_data['data']
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          if FLAGS.version_2_with_negative:
            is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length -
                                               1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
              tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
              continue
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = run_squad.SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

  return examples

def get_answer(data):
        
    eval_examples = get_squad_examples(data, is_training=False)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)

    run_squad.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=DOC_STRIDE,
        max_query_length=MAX_QUERY_LENGTH,
        is_training=False,
        output_fn=append_feature)

    global answer_model
    if answer_model == None:
        answer_model = Model(export_dir)
    
    all_results = answer_model.predict(eval_features=eval_features)
    pred = get_predicted_answer(eval_examples, eval_features, all_results,
                      N_BEST_SIZE, MAX_ANSWER_LENGTH,
                      DO_LOWER_CASE)
    temp = dict(pred.items())
    return temp['1']

context = "Camso Ltd. has developed an anti-static forklift tire that is engineered to resolve the safety issue of static electricity generated by non-marking tires. The accumulation of static electricity on forklifts is common in non-marking tires because of the silica used as reinforcing filler, allowing the tire to have isolating properties. This can lead to a number of problems like driver electrical shocks, forklift onboard electronic issues or outages and fire hazards. A patent filing assigned to Solideal Holding S.A. shows that static is dissipated by means of a column of carbon-black-reinforced rubber from an inner layer that protrudes through the outer non-marking layer. This column appears as small, round black area offset to one side of the tread face, allowing static to be discharged once every revolution."


class QA_Model(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        question = json_data['q']
        data = {'data':[{'title': 'Complaint', 'paragraphs':[{'context':context,'qas':[{'question':question,'id':"1"}]}]}]}
        answer = get_answer(data)
        return answer
    
    def get(self, question):
        data = {'data':[{'title': 'Complaint', 'paragraphs':[{'context':context,'qas':[{'question':question,'id':"1"}]}]}]}
        answer = get_answer(data)
        return answer
        
    
#api.add_resource(QA_Model, '/question')
api.add_resource(QA_Model, '/<string:question>')

if __name__ == '__main__':
    app.run(port=3131,host='0.0.0.0')