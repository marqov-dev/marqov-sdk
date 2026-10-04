# Offline SDK qualification only; not a deployable shared vendor profile.
FROM ghcr.io/astral-sh/uv@sha256:733b4042187702f832f7fdecb3aff14a61b288c4ca37af188bb5715c1caebaf8 AS uv
FROM compiler-runtime
USER 0:0
COPY --from=uv /uv /tmp/uv
COPY docs/qualification/ibm-python312-requirements.txt /tmp/requirements.txt
RUN /tmp/uv pip install --python /opt/venv/bin/python --require-hashes -r /tmp/requirements.txt
COPY marqov /tmp/sdk/marqov
COPY pyproject.toml README.md /tmp/sdk/
RUN /tmp/uv pip install --python /opt/venv/bin/python --no-deps /tmp/sdk && /tmp/uv pip check --python /opt/venv/bin/python && rm -rf /tmp/sdk /tmp/uv /tmp/requirements.txt
USER 10001:10001
