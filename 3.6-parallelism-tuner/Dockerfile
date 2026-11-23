FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m tuner
USER tuner
ENTRYPOINT ["ptuner"]
CMD ["search", "--model", "gpt3-13b", "--gpus", "8", "--batch", "512"]
