FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set environment variables
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal && \
    rm -rf /var/lib/apt/lists/*

# OpenCV with CUDA support
WORKDIR /opencv
RUN git clone https://github.com/opencv/opencv.git -b $CV_VERSION && \
    git clone https://github.com/opencv/opencv_contrib.git -b $CV_VERSION

# Apply fixes for OpenCV 4.2.0 to ensure CUDA support
RUN mkdir opencvfix && cd opencvfix && \
    git clone https://github.com/opencv/opencv.git -b 4.5.2 && \
    cd opencv/cmake && \
    cp -r FindCUDA /opencv/opencv/cmake/ && \
    cp FindCUDA.cmake /opencv/opencv/cmake/ && \
    cp FindCUDNN.cmake /opencv/opencv/cmake/ && \
    cp OpenCVDetectCUDA.cmake /opencv/opencv/cmake/

WORKDIR /opencv/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=$(which python2) \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc(); print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib(); print(get_python_lib())") \
    -D BUILD_opencv_python2=ON \
    -D BUILD_opencv_python3=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
    -D WITH_GSTREAMER=ON \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=8.6 \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /opencv

WORKDIR /
ENV OpenCV_DIR=/usr/share/OpenCV

# PyTorch for CUDA 11.3
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
ENV TORCH_CUDA_ARCH_LIST="8.6"

# OpenPCDet dependencies
RUN pip3 install numpy==1.23.0 kornia==0.6.5 llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d mayavi av2 pyquaternion

# Install spconv for CUDA 11.3
RUN pip3 install spconv-cu113

ENV NVIDIA_VISIBLE_DEVICES="all" \
    OpenCV_DIR=/usr/share/OpenCV \
    NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
    QT_GRAPHICSSYSTEM="native"
