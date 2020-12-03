import glob
import json
import os
import sys
from multiprocessing import Process

from create_pretraining_data import one_process

sys.path.insert(1, os.path.join(sys.path[0], ".."))


def make_tf_record(input_file, output_file, config):
    one_process(
        input_file,
        output_file,
        config["vocab"],
        config["max_seq_length"],
        config["do_lower_case"],
        config["dupe_factor"],
        config["short_seq_prob"],
        config["masked_lm_prob"],
        config["max_predictions_per_seq"],
        config["random_seed"],
        config["turn_sep"],
        config["for_multi"],
    )
    return True


if __name__ == "__main__":

    with open("./create_pretraining_data_config.json", "r") as f:
        config = json.load(f)

    input_prefix = config["input_prefix"]
    output_prefix = config["output_prefix"]
    num_split = config["num_split"]
    test_split = config["test_split"]
    test_split_rate = config["test_split_rate"]
    num_process = config["num_process"]

    if not input_prefix.endswith(".txt"):
        input_prefix = os.path.join(input_prefix, ".txt")

    if "{}_{}" not in output_prefix:
        output_prefix = output_prefix + "{}_{}"

    input_list = glob.glob(input_prefix)

    file_per_iter = num_process * num_split
    num_iter = int(len(input_list) / file_per_iter) + 1

    for i in range(num_iter):
        input_files = []
        output_files = []
        current_input = []
        current_output = []

        target_files = input_list[i * file_per_iter : (i + 1) * file_per_iter]

        if test_split:
            num_train = int(len(target_files) * (1 - test_split_rate))
        else:
            num_train = len(target_files)

        output_split = output_prefix.split("/")
        if output_split[-2] == "test":
            output_split[-2] = "train"
        output_prefix = "/".join(output_split)

        for j, target_file in enumerate(target_files):

            if j > num_train:
                output_split = output_prefix.split("/")
                if output_split[-2] == "train":
                    output_split[-2] = "test"
                output_prefix = "/".join(output_split)

            current_input.append(target_file)
            current_output.append(output_prefix.format(i, j))

            if (j + 1) % num_split == 0:
                input_files.append(current_input)
                output_files.append(current_output)
                current_input = []
                current_output = []

        if len(current_input) > 0 and len(current_output) > 0:
            input_files.append(current_input)
            output_files.append(current_output)

        print("*" * 10 + "Start Make TF record" + "*" * 10)
        print(f"Iteration : {i + 1}/{num_iter}")
        print(f"Number of File : {len(input_files) * num_split}")
        procs = []

        for input_file, output_file in zip(input_files, output_files):
            assert len(input_file) == len(output_file)
            proc = Process(target=make_tf_record, args=(input_file, output_file, config))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
