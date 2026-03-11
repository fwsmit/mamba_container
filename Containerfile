# For build
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y build-essential python3-dev git ninja-build cmake && \
    rm -rf /var/lib/apt/lists/*

# Install mamba and causal_conv1d
RUN git clone https://github.com/fwsmit/causal-conv1d.git /app/causal_conv1d
RUN cd /app/causal_conv1d && CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation -v .

RUN git clone https://github.com/fwsmit/mamba.git /app/mamba
RUN cd /app/mamba && MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation -v .

# Install some useful packages
RUN pip install pandas==2.3 numpy
RUN pip install onnx onnxscript onnxruntime
