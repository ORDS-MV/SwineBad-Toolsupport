FROM ubuntu:24.04

# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y \
        software-properties-common=0.99.49.1 \
        build-essential=12.10ubuntu1 \
        wget=1.21.4-1ubuntu4.1 \
        curl=8.5.0-2ubuntu10.6 \
        git=1:2.43.0-1ubuntu7.1 \
        python3=3.12.3-0ubuntu2 \
        python3-pip=24.0+dfsg-1ubuntu1.1  \
        python3-dev=3.12.3-0ubuntu2 \
        python3-venv=3.12.3-0ubuntu2 \
        libmagickwand-dev=8:6.9.12.98+dfsg1-5.2build2 \
        tesseract-ocr=5.3.4-1build5 \
        libtesseract-dev=5.3.4-1build5 && \
    rm -rf /var/lib/apt/lists/*

# Set python3 and pip3 as defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory
WORKDIR /usr/src/app

# Copy application code to working directory
COPY . ./

# Create and activate the virtual environment, then install dependencies
RUN python -m venv venv && \
    venv/bin/pip install --upgrade pip && \
    venv/bin/pip install -r requirements.txt && \
    venv/bin/pip uninstall -y opencv-python && \
    venv/bin/pip install opencv-python-headless==4.10.0.84 && \
    venv/bin/pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Set virtual environment python to be global python interpreter
ENV PATH="/usr/src/app/venv/bin:$PATH"

# Set entrypoint to be bash
ENTRYPOINT ["/bin/bash"]