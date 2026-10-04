FROM qristal-qpp-attribution:20260914 AS retained
FROM marqov-qristal-toolchain:qualification-20260909
COPY --from=retained /work/install-core /work/install-core
COPY --from=retained /work/install-xacc /work/install-xacc
RUN python3 -m pip install --no-cache-dir uv==0.11.7
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12.12 && uv venv /opt/py312 --python 3.12.12 && uv pip install --python /opt/py312/bin/python pybind11==2.13.6
COPY . /inputs
RUN cmake -S /inputs -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG' -DPython_EXECUTABLE=/opt/py312/bin/python -Dpybind11_DIR=$(/opt/py312/bin/python -m pybind11 --cmakedir) -DCMAKE_PREFIX_PATH='/work/install-core;/work/install-xacc' && cmake --build /build --parallel 1 && cmake --install /build
RUN mkdir /sdk && cp -R /inputs/sdk-marqov /sdk/marqov && cp /inputs/sdk-pyproject.toml /sdk/pyproject.toml && cp /inputs/sdk-README.md /sdk/README.md && uv pip install --python /opt/py312/bin/python /sdk 'qiskit>=2.0,<3' qiskit-qasm3-import pytest pytest-asyncio
ENV PYTHONPATH=/opt/python312-bindings
ENV LD_LIBRARY_PATH=/work/install-core/lib:/work/install-xacc/lib
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
USER 65532:65532
ENTRYPOINT ["/opt/py312/bin/python"]
