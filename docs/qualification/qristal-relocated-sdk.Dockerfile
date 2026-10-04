FROM ghcr.io/astral-sh/uv@sha256:733b4042187702f832f7fdecb3aff14a61b288c4ca37af188bb5715c1caebaf8 AS uv
FROM marqov-qristal-partner-runtime:full
USER 0:0
COPY --from=uv /uv /tmp/uv
COPY --from=sdk-source /marqov /tmp/sdk/marqov
COPY --from=sdk-source /pyproject.toml /README.md /tmp/sdk/
RUN /tmp/uv pip install --python /opt/venv/bin/python --no-deps /tmp/sdk && /tmp/uv pip check --python /opt/venv/bin/python && rm -rf /tmp/sdk /tmp/uv
USER 10001:10001
