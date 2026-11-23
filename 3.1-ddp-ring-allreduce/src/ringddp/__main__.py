import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: N workers = N cores, a fair comparison

from .cli import main  # noqa: E402

raise SystemExit(main())
