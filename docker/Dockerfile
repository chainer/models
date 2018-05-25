FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    gfortran \
    nasm \
    tmux \
    sudo \
    openssh-client \
    libgoogle-glog-dev \
    rsync \
    curl \
    wget \
    cmake \
    automake \
    libgmp3-dev \
    cpio \
    libtool \
    libyaml-dev \
    realpath \
    valgrind \
    software-properties-common \
    unzip \
    libz-dev \
    vim \
    emacs \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /root

# Install Intel MKL
RUN mkdir intel_mkl && cd intel_mkl && \
    wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12725/l_mkl_2018.2.199.tgz && \
    tar zxvf l_mkl_2018.2.199.tgz && rm -rf l_mkl_2018.2.199.tgz && \
    cd l_mkl_2018.2.199 && \
    sed -i -E "s/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g" silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf ${HOME}/intel_mkl
ENV LD_LIBRARY_PATH /opt/intel/mkl/lib/intel64:/opt/intel/lib/intel64:$LD_LIBRARY_PATH

# Install numpy & scipy with mkl backend
RUN echo "[mkl]\nlibrary_dirs = /opt/intel/mkl/lib/intel64\ninclude_dirs = /opt/intel/mkl/include\nmkl_libs = mkl_rt\nlapack_libs =" >> $HOME/.numpy-site.cfg && \
    pip3 install --no-binary :all: numpy==1.14.2 && \
    pip3 install --no-binary :all: scipy==1.0.1

# Install libjpeg-turbo
RUN mkdir libjpeg-turbo && cd libjpeg-turbo && \
    wget https://jaist.dl.sourceforge.net/project/libjpeg-turbo/1.5.1/libjpeg-turbo-1.5.1.tar.gz && \
    tar zxvf libjpeg-turbo-1.5.1.tar.gz && \
    rm -rf libjpeg-turbo-1.5.1.tar.gz && \
    cd libjpeg-turbo-1.5.1 && \
    ./configure --prefix=${HOME} && \
    make -j$(nproc) && \
    make install && \
    rm -rf ${HOME}/libjpeg-turbo

# Install OpenCV with libjpeg-turbo
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
    make install && \
    rm -rf ${HOME}/opencv

RUN pip3 install --no-cache-dir \
    ipython==6.3.1 \
    jupyter==1.0.0 \
    cython==0.28.2 \
    cupy-cuda90==4.0.0 \
    chainer==4.0.0 \
    matplotlib==2.2.2 \
    scikit-learn==0.19.1 \
    pandas==0.22.0

