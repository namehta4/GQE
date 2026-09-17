# ============================================================================
# ADAPT-GQE merged image: CUDA-Q (+cudaqx solvers) + LAMMPS (ML-IAP/MACE) +
# IBM Quantum (Qiskit) + this pipeline's Python dependencies, in one image.
#
# Built from your two working recipes (cudaq + LAMMPS), merged onto their
# shared parent `nersc/base_gpu:26.06`, with the following pruning decisions
# applied (see conversation history for the reasoning behind each):
#   - MACE-in-LAMMPS: via ML-IAP's Python "unified" interface, NOT a custom
#     pair_style mace plugin (neither source recipe built one). See the
#     VALIDATION WARNING below the LAMMPS build section -- this is the
#     highest-risk part of this merge, not independently verified here.
#   - Pruned entirely: DeepMD-kit, NEP_CPU, tensorflow[and-cuda].
#   - LAMMPS optional packages: pruned from ~50 to the minimal set ADAPT-GQE
#     actually exercises (REPLICA for NEB, plus basic organic-MD packages).
#     Dropped: QUIP, KIM, LATTE, ScaFaCoS, MSCG, PLUMED, SPIN, DPD-*,
#     MACHDYN, and ~25 other physics packages unrelated to gas-phase MD.
#   - Pruned: Sphinx docs toolchain, AWS SDK C++ / Braket support, QRMI
#     (Pasqal) prerequisite, qutip/qibo/qibojit/general-superstaq/qldpc.
#   - Kept + extended: qiskit, qiskit-nature, PLUS qiskit-ibm-runtime (new)
#     for actual IBM Quantum hardware submission.
#   - Standardized on ONE torch build (cu129, matching this CUDA 12.9.1-based
#     runtime) instead of the two source recipes' differing cu128/cu129 pins.
#
# UNVERIFIED IN THIS ENVIRONMENT: this Dockerfile has not been built. Two
# specific removals carry a "graceful auto-skip vs. hard build failure" risk
# that I could not check without your actual cuda-quantum/cudaqx fork
# sources -- see the comments at AWS SDK and QRMI below.
# ============================================================================

FROM docker.io/nersc/base_gpu:26.06
WORKDIR /opt
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# System packages (union of both source recipes' apt installs)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas-dev libgsl-dev liblapack-dev \
    git gfortran libtool automake patchelf doxygen \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# One shared Python venv for the whole image (both source recipes built
# their own copy of this at the same path -- merged into a single instance)
# ----------------------------------------------------------------------------
RUN python3 -m venv /opt/basic
ENV PATH=/opt/basic/bin:$PATH
RUN ln -sf /opt/basic/bin/pip /usr/local/bin/pip
ENV PYTHONPATH=/opt/basic/lib/python3.12/site-packages:$PYTHONPATH

RUN pip install --no-cache-dir --upgrade pip build ipython ipykernel

# Single torch build for the whole image: cu129, matching this CUDA
# 12.9.1-based runtime (the cudaq recipe's own justification for cu129 over
# the LAMMPS recipe's cu128 -- picking the more precise match since both
# LAMMPS's Kokkos/GPU build and mace-torch/cuequivariance only need a system
# CUDA toolkit + driver, not a specific torch wheel CUDA tag).
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu129

RUN pip install --no-cache-dir \
    cuequivariance-ops-torch-cu12 \
    cuequivariance-torch \
    cuequivariance \
    cupy-cuda12x

RUN pip install --no-cache-dir cython setuptools pyfftw
RUN pip install --no-cache-dir mpi4py -i https://pypi.anaconda.org/mpi4py/simple

# mace-torch: Python-side MACE-OFF (relaxation calls in mol_to_lammps_data.py,
# generate_reference_conformers.py, filter_neb_pathways.py) AND the source
# model our ML-IAP unified wrapper (docker build stage below) loads for the
# LAMMPS-side MD itself.
RUN pip install --no-cache-dir mace-torch==0.3.14

# General scientific/ML stack (cudaq recipe) -- kept: transformers is needed
# by model/, matplotlib by evaluation/, the rest is broadly useful and small.
RUN pip install --no-cache-dir \
    pandas seaborn numpy jupyterlab notebook lightning transformers \
    ml-collections astpretty scipy requests h5py torchdiffeq \
    scikit-build-core pytest matplotlib

# Chemistry + quantum-framework stack.
# Pruned vs. source recipe: qutip, general-superstaq, qldpc (not tied to
# CUDA-Q or IBM Quantum). Kept: pyscf/openfermion/openfermionpyscf/ffsim
# (needed directly by chemistry/build_hamiltonian.py and friends), qiskit +
# qiskit-nature (IBM Quantum). Added: qiskit-ibm-runtime, the actual package
# needed to submit jobs to IBM Quantum backends -- qiskit-nature alone only
# builds circuits/Hamiltonians, it doesn't talk to hardware.
RUN pip install --no-cache-dir \
    openfermionpyscf==0.4 \
    ffsim \
    pyscf \
    openfermion \
    qiskit \
    qiskit-nature \
    qiskit-ibm-runtime

# ADAPT-GQE-specific additions not in either source recipe.
RUN pip install --no-cache-dir rdkit ase pytket

# Sphinx docs toolchain: PRUNED (build-time only, not needed to run anything).

ENV PYTHONPATH=/opt/basic/lib/python3.12/site-packages:$PYTHONPATH

# ----------------------------------------------------------------------------
# cuTensor -- GPU tensor-contraction backend used by cuda-quantum's
# cuStateVec/cuTensorNet simulators.
# ----------------------------------------------------------------------------
ARG CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
ARG cutensor_archive=libcutensor-linux-x86_64-2.2.0.0-archive.tar.xz
RUN wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-2.2.0.0-archive.tar.xz \
    && mkdir -p "${CUTENSOR_INSTALL_PREFIX}" \
    && tar xf "${cutensor_archive}" --strip-components 1 -C "${CUTENSOR_INSTALL_PREFIX}" \
    && rm -rf "${cutensor_archive}"

# ----------------------------------------------------------------------------
# cuda-quantum: build prerequisites
# ----------------------------------------------------------------------------
RUN pip install --no-cache-dir "cmake>=3.27,<3.29" ninja auditwheel

RUN git clone --recurse-submodules https://github.com/namehta4/cuda-quantum.git /opt/cuda-quantum
WORKDIR /opt/cuda-quantum

ENV ZLIB_INSTALL_PREFIX=/usr/local/zlib
RUN ZLIB_VERSION=1.3.1 && \
    wget https://github.com/madler/zlib/releases/download/v${ZLIB_VERSION}/zlib-${ZLIB_VERSION}.tar.gz && \
    tar xzf zlib-${ZLIB_VERSION}.tar.gz && cd zlib-${ZLIB_VERSION} && \
    CFLAGS="-fPIC" ./configure --prefix=${ZLIB_INSTALL_PREFIX} --static && make && make install && \
    cd contrib/minizip && autoreconf --install && \
    CFLAGS="-fPIC" ./configure --prefix=${ZLIB_INSTALL_PREFIX} --disable-shared && make && make install && \
    cd /opt/cuda-quantum && rm -rf zlib-${ZLIB_VERSION} zlib-${ZLIB_VERSION}.tar.gz

ENV LLVM_INSTALL_PREFIX=/opt/llvm
ENV PYBIND11_INSTALL_PREFIX=/usr/local/pybind11
RUN bash scripts/build_llvm.sh -v

ENV BLAS_INSTALL_PREFIX=/usr/local/blas
RUN BLAS_VERSION=3.11.0 && \
    wget http://www.netlib.org/blas/blas-${BLAS_VERSION}.tgz && \
    tar xzf blas-${BLAS_VERSION}.tgz && cd BLAS-3.11.0 && \
    make FC=gfortran && \
    mkdir -p ${BLAS_INSTALL_PREFIX} && mv blas_*.a ${BLAS_INSTALL_PREFIX}/libblas.a && \
    cd /opt/cuda-quantum && rm -rf BLAS-3.11.0 blas-${BLAS_VERSION}.tgz

ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
RUN OPENSSL_VERSION=3.5.1 && PERL_VERSION=5.38.2 && \
    BUILDDIR=$(mktemp -d) && cd ${BUILDDIR} && \
    wget https://www.cpan.org/src/5.0/perl-${PERL_VERSION}.tar.gz && \
    tar xzf perl-${PERL_VERSION}.tar.gz && cd perl-${PERL_VERSION} && \
    ./Configure -des -Dprefix=${BUILDDIR}/perl5 && make && make install && \
    cd ${BUILDDIR} && \
    wget https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz && \
    tar xf openssl-${OPENSSL_VERSION}.tar.gz && cd openssl-${OPENSSL_VERSION} && \
    CFLAGS="-fPIC" CXXFLAGS="-fPIC" \
    ${BUILDDIR}/perl5/bin/perl Configure no-shared \
        --prefix=${OPENSSL_INSTALL_PREFIX} zlib \
        --with-zlib-include=${ZLIB_INSTALL_PREFIX}/include \
        --with-zlib-lib=${ZLIB_INSTALL_PREFIX}/lib && \
    make && make install && \
    rm -rf ${BUILDDIR}

ENV CURL_INSTALL_PREFIX=/usr/local/curl
RUN CURL_VERSION=8.5.0 && \
    BUILDDIR=$(mktemp -d) && cd ${BUILDDIR} && \
    wget https://curl.se/ca/cacert.pem && wget https://curl.se/ca/cacert.pem.sha256 && \
    mkdir -p ${CURL_INSTALL_PREFIX} && mv cacert.pem ${CURL_INSTALL_PREFIX}/ && \
    wget https://github.com/curl/curl/releases/download/curl-8_5_0/curl-${CURL_VERSION}.tar.gz && \
    tar xzf curl-${CURL_VERSION}.tar.gz && cd curl-${CURL_VERSION} && \
    cmake -G Ninja -B build \
        -DCMAKE_C_FLAGS="-fPIC" \
        -DCMAKE_INSTALL_PREFIX=${CURL_INSTALL_PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DCURL_USE_OPENSSL=ON \
        -DOPENSSL_ROOT_DIR=${OPENSSL_INSTALL_PREFIX} \
        -DCURL_ZLIB=ON \
        -DZLIB_ROOT=${ZLIB_INSTALL_PREFIX} \
        -DCURL_CA_BUNDLE=none \
        -DCURL_CA_PATH=none \
        -DCURL_USE_LIBSSH2=OFF -DCURL_USE_LIBPSL=OFF -DUSE_LIBIDN2=OFF \
        -DCURL_BROTLI=OFF -DCURL_ZSTD=OFF -DUSE_NGHTTP2=OFF -DENABLE_ARES=OFF \
        -DCURL_DISABLE_FTP=ON -DCURL_DISABLE_TFTP=ON -DCURL_DISABLE_SMTP=ON \
        -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON -DCURL_DISABLE_SMB=ON \
        -DCURL_DISABLE_GOPHER=ON -DCURL_DISABLE_TELNET=ON -DCURL_DISABLE_RTSP=ON \
        -DCURL_DISABLE_POP3=ON -DCURL_DISABLE_IMAP=ON -DCURL_DISABLE_FILE=ON \
        -DCURL_DISABLE_DICT=ON && \
    cmake --build build --config Release && \
    cmake --install build --config Release && \
    rm -rf ${BUILDDIR}

# AWS SDK C++ (Braket) -- PRUNED per your instruction. RISK: the cuda-quantum
# build script may either (a) auto-skip Braket support gracefully when the
# SDK isn't found at its expected prefix [most likely, if this is a
# find_package-gated optional feature], or (b) hard-fail expecting it
# unconditionally. Not verified against your fork. If the wheel build below
# fails referencing AWS/Braket/S3, either restore this block from your
# original recipe, or check pyproject.toml.cu12 / CMakeLists.txt for an
# explicit -DCUDAQ_ENABLE_AWS_BRAKET=OFF-style flag to add to the sed patches
# further down.

# QRMI (Pasqal connector) -- PRUNED. The pyproject.toml.cu12 already sets
# -DCUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=OFF before this Dockerfile's sed
# patches even run, so the connector should already be disabled regardless
# of whether libqrmi is present. Same caveat as AWS SDK above if this
# assumption is wrong for your fork.

# cuQuantum -- GPU-accelerated quantum simulation backends
ENV CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
RUN CUQUANTUM_VERSION=26.01.0.4 && \
    cuquantum_archive=cuquantum-linux-x86_64-${CUQUANTUM_VERSION}_cuda12-archive.tar.xz && \
    wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/${cuquantum_archive} && \
    mkdir -p ${CUQUANTUM_INSTALL_PREFIX} && \
    tar xf ${cuquantum_archive} --strip-components 1 -C ${CUQUANTUM_INSTALL_PREFIX} && \
    rm -rf ${cuquantum_archive}

# Build the cu12 wheel
ENV CUDA_QUANTUM_VERSION=0.0.0
ENV CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
RUN sed -i \
    's/cmake\.verbose = false/cmake.verbose = true/' pyproject.toml.cu12 && \
    sed -i \
    's/"-DCUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=OFF"/"-DCUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=OFF",\n    "-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF"/' pyproject.toml.cu12 && \
    sed -i \
    "s/'cupy-cuda12x ~= 13.6.0'/'cupy-cuda12x >= 13.6.0'/" pyproject.toml.cu12

RUN cp pyproject.toml.cu12 pyproject.toml && \
    PYTHON=/opt/basic/bin/python3 bash scripts/build_wheel.sh -c 12 -o /opt/cudaq-dist && \
    pip install --no-cache-dir /opt/cudaq-dist/cuda_quantum*.whl && \
    cmake -S /opt/cuda-quantum -B /opt/cuda-quantum/_skbuild \
        -U "Python*" -U "PYTHON*" \
        -DCUDAQ_DISABLE_CPP_FRONTEND=OFF \
        -DCUDAQ_DISABLE_TOOLS=OFF \
        -DPYTHON_EXECUTABLE=/opt/basic/bin/python3 \
        -DPython3_EXECUTABLE=/opt/basic/bin/python3 && \
    ninja -C /opt/cuda-quantum/_skbuild \
        'nvq++' cudaq-target-conf cudaq-quake cudaq-translate cudaq-opt fixup-linkage && \
    for tool in nvq++ cudaq-target-conf cudaq-quake cudaq-translate cudaq-opt fixup-linkage; do \
        [ -f /opt/cuda-quantum/_skbuild/bin/${tool} ] && \
        install -m 755 /opt/cuda-quantum/_skbuild/bin/${tool} \
            /opt/basic/lib/python3.12/site-packages/bin/${tool}; \
    done && \
    rm -rf /opt/cuda-quantum/_skbuild /opt/cuda-quantum/tpls /opt/cudaq-dist

# ----------------------------------------------------------------------------
# cudaqx (core;qec;solvers) -- solvers is what chemistry/run_adapt_vqe.py
# depends on (create_molecule/get_operator_pool/adapt_vqe). See earlier
# conversation: the PUBLIC NVIDIA/cudaqx repo dropped `solvers` from
# `libs/`; this build uses your fork, which apparently retains it.
# ----------------------------------------------------------------------------
WORKDIR /opt
ENV CUDAQ_DIR=/opt/basic/lib/python3.12/site-packages/lib/cmake/cudaq
ENV CUDAQX_INSTALL_PREFIX=/opt/nvidia/cudaqx

RUN git clone https://github.com/namehta4/cudaqx.git /opt/cudaqx

RUN cmake -G Ninja -S /opt/cudaqx -B /opt/cudaqx/build \
        -DCMAKE_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
        -DCUDAQ_DIR=${CUDAQ_DIR} \
        -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
        -DCUDAQX_ENABLE_LIBS="qec;solvers" \
        -DCUDAQX_BINDINGS_PYTHON=ON \
        -DCUDAQX_INCLUDE_DOCS=OFF \
        -DCUDAQX_INCLUDE_TESTS=OFF && \
    ninja -C /opt/cudaqx/build install && \
    rm -rf /opt/cudaqx/build

# libcudaq-em-default.so defines getRegisteredExecutionManager() which
# libcudaq-qec.so needs but which gets unloaded before cudaq_qec imports it.
RUN patchelf --add-needed libcudaq-em-default.so \
        /opt/nvidia/cudaqx/lib/libcudaq-qec.so

ENV PYTHONPATH=${CUDAQX_INSTALL_PREFIX}:${CUDAQX_INSTALL_PREFIX}/lib/python3.12/site-packages:$PYTHONPATH
ENV PATH=${CUDAQX_INSTALL_PREFIX}/bin:$PATH

# ----------------------------------------------------------------------------
# LAMMPS, pruned to ADAPT-GQE's minimal package needs.
#
# VALIDATION WARNING (highest risk in this Dockerfile): MACE is wired in via
# LAMMPS's ML-IAP "unified" Python interface (PKG_ML-IAP + MLIAP_ENABLE_PYTHON)
# rather than a dedicated pair_style mace plugin, per your instruction. This
# requires a custom MLIAPUnified wrapper class (lammps_md/mace_mliap_unified.py,
# separate deliverable) exposing a mace-torch model through LAMMPS's
# MLIAPUnified ABC. Neither this Dockerfile nor that wrapper has been built
# or run -- md_imipramine.in / neb_template.in's `pair_style mace` lines from
# earlier in this pipeline STILL NEED TO BE REWRITTEN to `pair_style mliap
# unified ...` to match. Do not treat the LAMMPS build succeeding as
# confirmation this integration works end-to-end.
#
# Pruned packages (vs. your original ~50): QUIP, KIM, LATTE, ScaFaCoS, MSCG,
# PLUMED, SPIN, DPD-*, MACHDYN, AMOEBA, BPM, BROWNIAN, CG-DNA, CG-SPICA,
# DIELECTRIC, ELECTRODE, MESONT, ORIENT, PLUGIN, REACTION, AMBER, CHARMM,
# COMPASS, DREIDING, GRANULAR, MC, ML-PACE, ML-SNAP, ML-POD, MISC-subpackages
# beyond MISC itself, POEMS, QEQ, SHOCK, SRD, H5MD, AWPMD, BOCS, DIFFRACTION,
# DRUDE, EFF, FEP, INTEL, MANIFOLD, MGPT, MOFFF, MOLFILE, PHONON, PTM, QMMM,
# QTB, SMTBQ, SPH, TALLY, UEF, YAFF, REAXFF, and DeepMD-kit/NEP_CPU entirely.
# ----------------------------------------------------------------------------
WORKDIR /opt
RUN git clone -b stable_22Jul2025_update4 --recursive https://github.com/lammps/lammps.git lammps
WORKDIR /opt/lammps
RUN mkdir build
WORKDIR /opt/lammps/build

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/compat
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/basic/lib/python3.12/site-packages/torch/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/basic/lib/python3.12/site-packages/nvidia/nccl/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/fftw/install/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/blas
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/fftw

# Kokkos bug workaround, unrelated to which packages are enabled -- harmless
# to keep even though this fix targets an electron-stopping fix module we
# aren't using; kept from your original recipe.
RUN sed -i '/tag;/s/t_int_1d_randomread/t_tagint_1d_randomread/' /opt/lammps/src/KOKKOS/fix_electron_stopping_kokkos.h

RUN cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_CXX_COMPILER=/opt/lammps/lib/kokkos/bin/nvcc_wrapper -D LAMMPS_EXCEPTIONS=ON \
        -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D CMAKE_CXX_STANDARD=17 \
        -D CMAKE_POSITION_INDEPENDENT_CODE=ON -D LAMMPS_SIZES=BIGBIG \
        -D PKG_KOKKOS=yes -D Kokkos_ARCH_ZEN3=yes -D Kokkos_ARCH_AMPERE80=ON \
        -D Kokkos_ENABLE_CUDA=yes -D Kokkos_ENABLE_OPENMP=yes \
        -D PKG_GPU=yes -D GPU_API=cuda -D GPU_ARCH=sm_80 \
        -D PKG_REPLICA=yes \
        -D PKG_MOLECULE=yes -D PKG_KSPACE=yes -D PKG_RIGID=yes -D PKG_MANYBODY=yes \
        -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-PAIR=yes -D PKG_EXTRA-COMPUTE=yes \
        -D PKG_EXTRA-MOLECULE=yes -D PKG_MISC=yes \
        -D PKG_PYTHON=yes -D PKG_ML-IAP=yes -D MLIAP_ENABLE_PYTHON=yes \
        -D PKG_OPENMP=yes \
        -D FFT=FFTW3 -D FFT_PACK=array -D FFT_KOKKOS=CUFFT \
        -D LAMMPS_INSTALL_RPATH=ON ../cmake
RUN make -j 4
RUN make install

WORKDIR /opt/lammps/python
RUN pip install --no-cache-dir -U build wheel setuptools
RUN pip wheel .
RUN pip install --no-cache-dir --force-reinstall lammps-*.whl
WORKDIR /opt

ENV PATH=/opt/lammps/install/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/lammps/install/lib

# ----------------------------------------------------------------------------
# Housekeeping: strip debug symbols, clear caches (from the cudaq recipe,
# extended to cover the LAMMPS install too).
# ----------------------------------------------------------------------------
RUN find /opt/basic/lib/python3.12/site-packages/lib \
         /opt/nvidia/cudaqx/lib \
         /opt/nvidia/cuquantum/lib \
         /opt/lammps/install/lib \
         -name "*.so*" -type f \
         -exec strip --strip-debug {} \; 2>/dev/null || true && \
    find /opt/llvm/lib -name "*.so*" -type f \
         -exec strip --strip-debug {} \; 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

RUN pip install --no-cache-dir "ipie @ git+https://github.com/rr637/ipie.git@develop"
RUN pip install --no-cache-dir --upgrade "jax[cuda12]"

# ----------------------------------------------------------------------------
# ADAPT-GQE pipeline code
# ----------------------------------------------------------------------------
COPY lammps_md /opt/adapt-gqe/lammps_md
COPY chemistry /opt/adapt-gqe/chemistry
COPY model /opt/adapt-gqe/model
COPY evaluation /opt/adapt-gqe/evaluation
COPY hardware /opt/adapt-gqe/hardware
ENV PYTHONPATH=/opt/adapt-gqe:$PYTHONPATH

WORKDIR /opt/adapt-gqe
