FROM nvidia/cuda:10.2-base-ubuntu18.04

# # Deal with pesky Python 3 encoding issue
# ENV LANG C.UTF-8 
# ARG DEBIAN_FRONTEND=noninteractive

# # Global Path Setting
ENV CUDA_HOME /usr/local/cuda-10.2
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

# apt Install
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip

# Install related packages
RUN pip3 install \
    tqdm \
    numpy \ 
    pytz pandas python-dateutil \
    scipy threadpoolctl joblib scikit-learn

# Install pytorch
# typing-extensions, torch, pillow, torchvision, torchaudio
RUN pip3 install \
    torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# System Cleanup
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

# COPY data directory
COPY ./data /home/data
COPY ./CaptchaOCR /home/CaptchaOCR

WORKDIR /home/CaptchaOCR
VOLUME /home/CaptchaOCR

ENTRYPOINT ["/bin/bash"]
