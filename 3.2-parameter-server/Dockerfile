# Stage 1: build the Go shard server (static binary, standard library only)
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/psserver ./cmd/psserver

# Stage 2: one image for shards and workers
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.* && useradd -u 10001 -m ps
WORKDIR /app
COPY --from=build /out/psserver /usr/local/bin/psserver
COPY pskit ./pskit
ENV PSSERVER=/usr/local/bin/psserver PYTHONUNBUFFERED=1
USER 10001
EXPOSE 7000
ENTRYPOINT ["psserver"]
CMD ["-addr=:7000"]
