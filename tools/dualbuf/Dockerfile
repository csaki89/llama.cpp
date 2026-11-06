# NAS image (TrueNAS, Docker, AMD RX 6700 = gfx1031): HIP build of the dual-buffer branch.
# Build with tools/dualbuf/build-nas.sh. Binaries: /opt/llama.cpp/build/bin (llama-completion, llama-server, ...).
ARG ROCM_TAG=7.0-complete
FROM rocm/dev-ubuntu-24.04:${ROCM_TAG}

# gfx1030 code runs on the gfx1031 card with HSA_OVERRIDE_GFX_VERSION=10.3.0
ARG GPU_TARGET=gfx1030
ARG REPO_URL=https://github.com/csaki89/llama.cpp.git
ARG BRANCH=dual-buffer
# change to force a fresh clone (docker layer cache)
ARG CACHE_BUST=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*

# optional BuildKit secret gh_token for a private fork: --secret id=gh_token,env=GH_TOKEN
RUN --mount=type=secret,id=gh_token \
    echo "bust=${CACHE_BUST}" && \
    if [ -s /run/secrets/gh_token ]; then \
        git clone --depth 1 -b "${BRANCH}" "https://x-access-token:$(cat /run/secrets/gh_token)@${REPO_URL#https://}" /opt/llama.cpp; \
        git -C /opt/llama.cpp remote set-url origin "${REPO_URL}"; \
    else \
        git clone --depth 1 -b "${BRANCH}" "${REPO_URL}" /opt/llama.cpp; \
    fi

WORKDIR /opt/llama.cpp
# Xeon E5-2673 v4 (Broadwell): AVX2/FMA/F16C/BMI2, no AVX512; AVX2 is needed for the Q4_K repack
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON -DAMDGPU_TARGETS=${GPU_TARGET} \
        -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    && cmake --build build -j "$(nproc)" \
        --target llama-completion llama-server llama-cli llama-bench

ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PATH=/opt/llama.cpp/build/bin:/opt/llama.cpp/tools/dualbuf:${PATH}
# models: /models (Ollama blobs, read-only), results: /work
WORKDIR /work
CMD ["bash"]
