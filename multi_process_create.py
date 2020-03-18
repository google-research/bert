# coding: utf-8
import multiprocessing
import os

root = '/data/nfs/wuwei/data'
out = '/data/nfs/wuwei/tfrecord'


def func(txt_name):
    print(F"Processing {txt_name}")
    txt_path = os.path.join(root, txt_name)
    out_path = os.path.join(out, txt_name.replace('.txt', '.tfrecord'))
    cmd = F'/data/nfsdata2/home/wuwei/.conda/envs/allennlp/bin/python /data/nfsdata2/home/wuwei/study/bert/create_pretraining_data.py --input_file {txt_path} --output_file {out_path} --vocab_file /data/nfsdata/nlp/BERT_BASE_DIR/chinese_L-12_H-768_A-12/vocab.txt'
    if not os.path.exists(out_path):
        os.system(cmd)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=100)
    for fname in os.listdir(root):
        if fname.endswith('.txt'):
            pool.apply_async(func, (fname,))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
