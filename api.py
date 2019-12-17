from flask import Flask, jsonify
from flask import request
from build_data import build_rule_kcs_dict, load_kcs_case_dict, build_case_pairs
from run_classifier import load_model, do_predict, flags, tf
import json
import os

app = Flask(__name__)

se_customer_tasks = [
    {
        'customer_input_id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'rule_list': ['', ''],
        'done': False
    },
    {
        'customer_input_id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

estimator = load_model()


def handle_rule(rule_list):
    pass


def get_filter_rank_rules(result_df):
    kcs_rule_dict = json.load(open("./data/kcs_rule_dict.json"))
    result_df['predict_label'] = result_df.apply(lambda x: 1 if x['predict_value'] >= 0.5 else 0, axis=1)
    filter_df = result_df.groupby(['caseb_kcs'])['predict_value'].agg(['mean']).reset_index()
    # filter_df[filter_df['mean'] > 0.2]
    filter_df['rule'] = filter_df.apply(lambda x: kcs_rule_dict[x['caseb_kcs']], axis=1)
    filter_df = filter_df[['rule', 'mean']]
    filter_df.columns = ['rule', 'score']
    filter_df = filter_df.sort_values(by=['score'], ascending=False)
    return filter_df


@app.route('/se_customer_tasks/api/customer_inputs', methods=['GET'])
def get_tasks():
    return jsonify({'customer_inputs': se_customer_tasks})


@app.route('/se_customer_tasks/api/customer_inputs', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
        abort(400)
    task = {
        'customer_input_id': se_customer_tasks[-1]['customer_input_id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'hit_rules': request.json.get('hit_rules', []),
        'done': False
    }
    se_customer_tasks.append(task)
    rules = task['hit_rules']
    customer_input = task['title']
    print(rules)
    print(customer_input)
    rk_dict = build_rule_kcs_dict()
    kcs_cases_df = load_kcs_case_dict()
    test_df = build_case_pairs(rk_dict, kcs_cases_df, customer_input, rules)
    print(test_df)
    result_df = do_predict(estimator, test_df)
    result_df = get_filter_rank_rules(result_df)
    print(result_df)
    return jsonify(result_df.to_json(orient='records')), 201


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    app.run(debug=True)
