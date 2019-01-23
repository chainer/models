FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
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
    wget \
    cmake \
    cmake-curses-gui \
    automake \
    libgmp3-dev \
    cpio \
    libtool \
    libyaml-dev \
    valgrind \
    software-properties-common \
    unzip \
    libz-dev \
    vim \
    apt-utils \
    emacs \
    zsh \
    locales \
    ruby \
    htop \
    libeigen3-dev \
    python \
    python-dev \
    python-pip \
    python-pip \
    python-wheel \
    python-setuptools \
    python-numpy \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3-numpy \
    libibverbs-dev \
    libibumad-dev \
    libmlx4-1 \
    infiniband-diags \
    ibverbs-utils \
    perftest && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ARG GROUP_ID=51000
RUN addgroup --gid ${GROUP_ID} fulltime

# Install OpenMPI with cuda & verbs
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
RUN mkdir /root/lib && cd /root/lib && \
    curl -L -O https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.1.tar.gz && \
    tar zxvf openmpi-3.1.1.tar.gz && rm -rf openmpi-3.1.1.tar.gz && \
    cd openmpi-3.1.1 && \
    ./configure --with-cuda --with-verbs && \
    make -j8 && make install && \
    ompi_info --parsable --all | grep -q "mpi_built_with_cuda_support:value:true"

ARG USER_ID=1000
ARG USER_NAME=ubuntu
RUN mkdir -p /home/${USER_NAME}
RUN useradd -d /home/${USER_NAME} -g ${GROUP_ID} -G sudo -u ${USER_ID} ${USER_NAME}
RUN addgroup --gid ${USER_ID} ${USER_NAME}
RUN chown ${USER_NAME}:root /home/${USER_NAME}
RUN chsh -s /usr/bin/zsh ${USER_NAME}

# Switch to USER_NAME
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
ENV HOME /home/${USER_NAME}

# Install oh-my-zsh
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh && \
    cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc && \
    echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)"

# Install miniconda3-4.3.0
RUN pyenv install miniconda3-4.3.30 && \
    pyenv global miniconda3-4.3.30 && \
    pyenv rehash

RUN CONDA=$(pyenv which conda) && \
    PIP=$(pyenv which pip) && \
    $PIP install -U pip && \
    $PIP install PyHamcrest && \
    $CONDA install -y numpy scipy scikit-learn scikit-image jupyter matplotlib cython protobuf pandas h5py cmake

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib
ENV LIBRARY_PATH=$LIBRARY_PATH:$HOME/lib
ENV CPATH=$CPATH:/usr/local/include:$HOME/include
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:$HOME/lib' >> ~/.zshrc && \
    echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:$HOME/lib' >> ~/.zshrc && \
    echo 'export CPATH=$CPATH:/usr/local/include:$HOME/include' >> ~/.zshrc

# Install libjpeg-turbo
RUN if [ ! -d $HOME/lib ]; then mkdir $HOME/lib; fi
RUN cd $HOME/lib && mkdir libjpeg-turbo && cd libjpeg-turbo && \
    curl -L -O https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.0.tar.gz && \
    tar zxvf 2.0.0.tar.gz && \
    rm -rf 2.0.0.tar.gz && \
    cd libjpeg-turbo-2.0.0 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${HOME} .. && \
    make -j$(nproc) && \
    make install && \
    cd ${HOME} && rm -rf ${HOME}/libjpeg-turbo

# Install opencv
RUN cd $HOME/lib && mkdir opencv && cd opencv && \
    curl -L -O https://github.com/opencv/opencv/archive/3.4.3.zip && \
    unzip 3.4.3.zip && rm -rf 3.4.3.zip && \
    curl -L -O https://github.com/opencv/opencv_contrib/archive/3.4.3.zip && \
    unzip 3.4.3.zip && rm -rf 3.4.3.zip && \
    cd opencv-3.4.3 && mkdir build && cd build

RUN cd $HOME/lib/opencv/opencv-3.4.3/build && \
    PYTHON=$(pyenv which python) && \
    $(pyenv which cmake) \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=${HOME} \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_QT=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_CUBLAS=OFF \
    -DWITH_CUFFT=OFF \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=OFF \
    -DBUILD_TIFF=ON \
    -DWITH_TIFF=OFF \
    -DBUILD_PNG=ON \
    -DWITH_PNG=OFF \
    -DJPEG_INCLUDE_DIR=${HOME}/include \
    -DJPEG_LIBRARY=${HOME}/lib/libjpeg.so \
    -DOPENCV_EXTRA_MODULES_PATH=${HOME}/lib/opencv/opencv_contrib-3.4.3/modules \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=$(which $PYTHON) \
    -DPYTHON3_INCLUDE_DIR=$($PYTHON -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
    -DPYTHON3_INCLUDE_DIR2t=$($PYTHON -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=$($PYTHON -c 'import numpy; print(numpy.get_include())') \
    -DPYTHON3_LIBRARY=find $(pyenv prefix)/lib -name "libpython*.so" \
    .. && \
    LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:$HOME/lib make -j$(nproc) && make install

# Install MPI4PY
RUN PIP=$(pyenv which pip) && \
    $PIP install mpi4py

ARG CHAINER_VERSION=6.0.0b1
ARG CUPY_VERSION=6.0.0b1
ARG CHAINERCV_VERSION=0.11.0

# Install Chainer family
RUN PIP=$(pyenv which pip) && \
    $PIP install chainer==$CHAINER_VERSION && \
    $PIP install cupy-cuda92==$CUPY_VERSION && \
    $PIP install chainercv==$CHAINERCV_VERSION

# Install other python packages
RUN PIP=$(pyenv which pip) && \
    $PIP install jupyterthemes && \
    JT=$(pyenv which jt) && \
    $JT -f dejavu -T -N && \
    $PIP install xlrd && \
    $PIP install imageio && \
    $PIP install tqdm && \
    $PIP install pyyaml && \
    $PIP install ipdb && \
    $PIP install pynvvl-cuda92 --pre

# Set environment variable for OpenCV python wrapper
ENV PYTHONPATH=$HOME/lib/python3.6/site-packages:$PYTHONPATH

# Install Linuxbrew
ENV PATH $PATH:$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin
RUN git clone https://github.com/Linuxbrew/brew.git ~/.linuxbrew/Homebrew \
    && mkdir ~/.linuxbrew/bin \
	&& ln -s ~/.linuxbrew/Homebrew/bin/brew ~/.linuxbrew/bin/ \
	&& brew config
RUN echo "export PATH=$PATH:'$(brew --prefix)/bin:$(brew --prefix)/sbin'" >> ~/.zshrc

