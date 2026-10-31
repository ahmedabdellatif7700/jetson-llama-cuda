
# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

# Disable CUDA forward-compatibility libraries on Jetson.
# The NVIDIA container runtime supplies the host CUDA driver.
# This avoids the CDI enable-cuda-compat crash in toolkit 1.19.1.
RUN for dir in \
        /usr/local/cuda-13.2/compat \
        /usr/local/cuda-13.2/compat_orin; do \
        if [ -d "$dir" ]; then \
            mv "$dir" "$dir.disabled"; \
        fi; \
    done

# Install build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        ccache \
        libcurl4-openssl-dev \
        libssl-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# CUDA driver stub is used for linking only.
# Use the CUDA linking stub supplied by the base image.
RUN mkdir -p /opt/cuda-stubs && \
    ln -s /usr/local/cuda-13.2/targets/sbsa-linux/lib/stubs/libcuda.so \
        /opt/cuda-stubs/libcuda.so && \
    ln -s libcuda.so /opt/cuda-stubs/libcuda.so.1


# Pin llama.cpp to the previously tested revision.
ARG LLAMA_CPP_COMMIT=5047748

WORKDIR /opt

RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git checkout "${LLAMA_CPP_COMMIT}"

WORKDIR /opt/llama.cpp

# Build llama.cpp with CUDA support for SM 8.7.
RUN --mount=type=cache,id=orin-llama-build-132,target=/opt/llama.cpp/build,sharing=locked \
    --mount=type=cache,id=orin-llama-ccache,target=/root/.cache/ccache,sharing=locked \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DGGML_CCACHE=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/opt/cuda-stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/cuda-stubs" && \
    cmake --build build --target llama-server -j 2 && \
    mkdir -p /opt/llama-install && \
    cp -a build/bin/. /opt/llama-install/

# curl is used by the compose healthcheck. Installed after the build so
# adding it does not invalidate the cached llama.cpp layers.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Runtime configuration.
ENV LD_LIBRARY_PATH=/opt/llama-install

EXPOSE 8080

ENTRYPOINT ["/opt/llama-install/llama-server"]

