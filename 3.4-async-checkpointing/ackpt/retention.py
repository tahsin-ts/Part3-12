"""Retention policy and garbage collection.

Rules: keep the newest `keep_last` committed checkpoints; delete older ones MANIFEST FIRST
(so a half-deleted checkpoint is never mistaken for a valid one); delete files of uncommitted
steps older than the newest commit (debris of crashed saves). Never touch `protect`.
"""
from .manifest import MANIFEST, parse_step, step_dir


def plan(keys, keep_last, protect=None):
    by_step = {}
    for k in keys:
        s = parse_step(k)
        if s is not None:
            by_step.setdefault(s, []).append(k)
    committed = sorted(s for s, ks in by_step.items() if any(k.endswith("/" + MANIFEST) for k in ks))
    keep = set(committed[-keep_last:]) if keep_last > 0 else set(committed)
    newest = committed[-1] if committed else -1
    delete = []
    for s in sorted(by_step):
        if s in keep or s == protect:
            continue
        if s in committed or s < newest:       # old checkpoint, or crash debris below the newest commit
            delete.append(s)
    return delete, by_step


def gc(storage, keep_last, protect=None):
    delete, by_step = plan(storage.list_all(), keep_last, protect)
    for s in delete:
        man = f"{step_dir(s)}/{MANIFEST}"
        if man in by_step[s]:
            storage.delete(man)                 # 1. un-commit
        for k in by_step[s]:
            if k != man:
                storage.delete(k)               # 2. then the data
    return delete
