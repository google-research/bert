ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.08-py3

FROM tensorrtserver_client as trt

FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y pbzip2 pv bzip2

RUN pip install toposort networkx pytest nltk tqdm html2text progressbar

WORKDIR /workspace
RUN git clone https://github.com/openai/gradient-checkpointing.git
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git
RUN git clone https://github.com/titipata/pubmed_parser

RUN pip3 install /workspace/pubmed_parser

#Copy the perf_client over
COPY --from=trt /workspace/build/perf_client /workspace/build/perf_client

#Copy the python wheel and install with pip
COPY --from=trt /workspace/build/dist/dist/tensorrtserver*.whl /tmp/
RUN pip install /tmp/tensorrtserver*.whl && rm /tmp/tensorrtserver*.whl

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH /workspace/bert
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
