# AI Infrastructure & ML Systems Engineering Projects (Parts 3-12)

Hands-on, tested implementations of distributed training, LLM inference and serving, and ML platform systems. Each folder is a self-contained project with its own README, architecture diagram, source code, tests, benchmarks, Dockerfile and CI workflow (CI runs only when that folder changes).

> **About GPU numbers:** the projects were built and measured on CPU-only machines. Wherever a GPU result is shown it is clearly labelled **PROJECTED**, with its assumptions and the command to measure it on real hardware.

## Part 3: Distributed Training Systems

| # | Project | Level | Language | Summary |
|---|---|---|---|---|
| 3.1 | [Data-Parallel Training from Scratch with Ring All-Reduce](./3.1-ddp-ring-allreduce) | Intermediate | Python | A small, complete implementation of distributed data-parallel (DDP) training in pure Python and NumPy. |
| 3.2 | [A Parameter Server](./3.2-parameter-server) | Intermediate | Go, Python | A working parameter server for sparse models: model weights are split across Go shard servers with consistent hashing, |
| 3.3 | [High-Throughput Streaming Data Loader](./3.3-streaming-data-loader) | Intermediate | Python | (S3-style range requests). It decodes in worker processes, prefetches into bounded queues, shuffles with a |
| 3.4 | [Asynchronous Distributed Checkpointing Library](./3.4-async-checkpointing) | Intermediate | Python | A small, complete checkpointing library for sharded training state. Each rank saves its own shard of the |
| 3.5 | [Fault-Tolerant Elastic Training](./3.5-elastic-training) | Advanced | Python | slow down or join late. It is a small, readable re-implementation of the ideas behind PyTorch's |
| 3.6 | [Automatic Parallelism Configuration Tuner](./3.6-parallelism-tuner) | Advanced | Python | over these GPUs?** It enumerates every combination of data parallelism (DP), tensor parallelism (TP), pipeline |
| 3.7 | [70B-Parameter Training with 3D Parallelism](./3.7-3d-parallelism-70b) | Pro | Python | using one operating-system process per "GPU" and plain NumPy, and proves that the result is the same model a |

## Repository layout

```
part3-12/
├── <part>.<n>-<project>/   one folder per project (README, src, tests, docs, Dockerfile, deploy)
├── .github/workflows/      one workflow per project, filtered by path
└── LICENSE
```

## Running a project

```bash
git clone https://github.com/tahsin-ts/part3-12
cd part3-12/<project-folder>
make test     # run the tests
make demo     # see it work
```

## Related repositories

- [dataset-lineage](https://github.com/tahsin-ts/dataset-lineage) · [tokenizer-service](https://github.com/tahsin-ts/tokenizer-service) · [cuda-kernels](https://github.com/tahsin-ts/cuda-kernels) · [triton-flash-attenti0n](https://github.com/tahsin-ts/triton-flash-attenti0n) · [gpu-memory-profiler](https://github.com/tahsin-ts/gpu-memory-profiler) · [nccl-topology-bench](https://github.com/tahsin-ts/nccl-topology-bench) · [llm-gateway](https://github.com/tahsin-ts/llm-gateway)

## License

MIT. Copyright (c) 2026 Tahsin Shahrear. See [LICENSE](./LICENSE).
