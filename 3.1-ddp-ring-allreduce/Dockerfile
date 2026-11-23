FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -u 1000 -m ddp
USER ddp
EXPOSE 29500
ENTRYPOINT ["ringddp"]
CMD ["train", "--workers", "2", "--epochs", "2"]
