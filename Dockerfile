FROM nvidia/cuda:10.2-base-ubuntu18.04

# apt Install
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip \
    git

# Install related packages
RUN pip3 install \
    tqdm \
    numpy \
    pandas \
    # pytz pandas python-dateutil 
    scikit-learn \
    # scipy threadpoolctl joblib scikit-learn
    tensorboard
    # absl-py cachetools certifi chardet google-auth google-auth-oauthlib grpcio \
    # importlib-metadata markdown oauthlib protobuf pyasn1 pyasn1-modules requests requests-oauthlib rsa \
    # setuptools tensorboard tensorboard-data-server tensorboard-plugin-wit urllib3 werkzeug zipp

# Install pytorch - typing-extensions, torch, pillow, torchvision, torchaudio
RUN pip3 install \
    torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# System Cleanup
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /home
VOLUME /home

ENTRYPOINT ["/bin/bash"]
