# Use an official centos7 image
FROM centos:7

# gcc because we need regex and pyldap
# openldap-devel because we need pyldap
RUN yum update -y \
    && yum install -y https://centos7.iuscommunity.org/ius-release.rpm \
    && yum install -y python36u python36u-libs python36u-devel python36u-pip \
    && yum install -y which gcc \
    && yum install -y openldap-devel

# pipenv installation
RUN pip3.6 install pipenv
RUN ln -s /usr/bin/pip3.6 /bin/pip
RUN rm /usr/bin/python
# python must be pointing to python3.6
RUN ln -s /usr/bin/python3.6 /usr/bin/python

MAINTAINER Leonardo Apolonio
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
ADD asset/vocab.txt asset/vocab.txt
CMD ["python", "run_app.py"]
