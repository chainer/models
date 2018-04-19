FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    nasm \
    tmux \
    sudo \
    openssh-client \
    rsync \
    libgoogle-glog-dev \
    curl \
    cmake \
    automake \
    libgmp3-dev \
    libtool \
    libyaml-dev \
    realpath \
    wget \
    valgrind \
    software-properties-common \
    unzip \
    libz-dev \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir \
    numpy \
    ipython \
    jupyter \
    cython \
    cupy-cuda91==4.0.0 \
    chainer==4.0.0

WORKDIR /root

RUN mkdir libjpeg-turbo && cd libjpeg-turbo && \
    wget https://jaist.dl.sourceforge.net/project/libjpeg-turbo/1.5.1/libjpeg-turbo-1.5.1.tar.gz && \
    tar zxvf libjpeg-turbo-1.5.1.tar.gz && \
    rm -rf libjpeg-turbo-1.5.1.tar.gz && \
    cd libjpeg-turbo-1.5.1 && \
    ./configure --prefix=${HOME} && \
    make -j$(nproc) && \
    make install && \
    rm -rf ${HOME}/libjpeg-turbo

RUN mkdir opencv && cd opencv && \
    wget https://github.com/opencv/opencv/archive/3.4.1.tar.gz && \
    tar zxvf 3.4.1.tar.gz && rm -rf 3.4.1.tar.gz && \
    wget https://github.com/opencv/opencv_contrib/archive/3.4.1.tar.gz && \
    tar zxvf 3.4.1.tar.gz && rm -rf 3.4.1.tar.gz && \
    mkdir build && cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_QT=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN=6.0 \
    -DCUDA_ARCH_PTX= \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=OFF \
    -DJPEG_INCLUDE_DIR=${HOME}/include \
    -DJPEG_LIBRARY=${HOME}/lib/libjpeg.so \
    -DOPENCV_EXTRA_MODULES_PATH=${HOME}/opencv/opencv_contrib-3.4.1/modules \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=$(which python3) \
    -DPYTHON3_INCLUDE_DIR=$(python3 -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c 'import numpy; print(numpy.get_include())') \
    -DPYTHON3_LIBRARIES=$(find /usr/lib -name 'libpython*.so') \
    ${HOME}/opencv/opencv-3.4.1 && \
    make -j$(nproc) && \
    make install

