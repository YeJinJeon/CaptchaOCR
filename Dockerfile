FROM nvidia/cuda:10.2-base-ubuntu18.04

# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8 
ARG DEBIAN_FRONTEND=noninteractive

# Global Path Setting
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/lib

# Install essential Ubuntu packages
RUN apt-get update && apt-get install -y \
    # Pillow, opencv
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-dev \
    # essential
    wget \
    git \
    vim 

# Install Miniconda and Python 3.8
ENV PATH /opt/conda/bin:$PATH
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
 && wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh \
 && chmod +x ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya   

# Intall pytorch for CUDA 10.2
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.6.0=py38_cu102" \
 && conda clean -ya

# Install related python library
RUN conda install mkl numpy scipy scikit-learn pandas matplotlib tensorboard && \
    conda install -c conda-forge opencv tqdm 
    
# System Cleanup
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

VOLUME /home/CaptchaOCR
WORKDIR /home/CaptchaOCR

# IPython
ENTRYPOINT ["/bin/bash"]