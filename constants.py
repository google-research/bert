INPUT_RAW_DATA_FILE = 'data/input_file_long.csv'
VOCAB_FILE = 'data/bert_pretrained/vocab.txt'
OUTPUT_PRETRAINING_DATA = 'data/model_input_data_long'
DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 128

BERT_CONFIG_FILE = 'data/bert_pretrained/bert_config.json'
MODEL_OUTPUT_DIR = 'data/model_output_long/'
BERT_INIT_CHECKPOINT = 'data/bert_pretrained/bert_model.ckpt'
DO_TRAIN = True
DO_EVAL = True
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_TRAIN_STEPS = 100000
NUM_WARMUP_STEPS = 10000
SAVE_CHECKPOINT_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
MAX_EVAL_STEPS = 100

