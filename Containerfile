# For build
FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-devel

WORKDIR /app
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y build-essential python3-dev git ninja-build cmake && \
    rm -rf /var/lib/apt/lists/*

# Install mamba and causal_conv1d
RUN git clone https://github.com/fwsmit/causal-conv1d.git /app/causal_conv1d
RUN cd /app/causal_conv1d && CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --break-system-packages -v .

RUN git clone https://github.com/fwsmit/mamba.git /app/mamba
RUN cd /app/mamba && MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation --break-system-packages -v .

# Install some useful packages
RUN pip install --break-system-packages pandas==2.3 numpy
RUN pip install --break-system-packages onnx onnxscript onnxruntime onnxoptimizer

# temporary fix
RUN pip install --break-system-packages transformers==5.3.0
