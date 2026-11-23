FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pytest
COPY ackpt ./ackpt
COPY tests ./tests
COPY pyproject.toml ./
RUN pip install --no-cache-dir . && useradd -m ckpt && mkdir -p /ckpt && chown ckpt /ckpt
USER ckpt
VOLUME /ckpt
ENTRYPOINT ["ackpt"]
CMD ["demo", "--dir", "/ckpt"]
