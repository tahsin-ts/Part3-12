"""Read one shard from any store: footer -> index -> records, streamed in large coalesced range reads."""
from .shards import FOOTER, REC, CorruptRecord, check_record, parse_footer, parse_index


class ShardReader:
    def __init__(self, store, name, block_bytes=1 << 20, on_corrupt="raise", size=None, tail_bytes=1 << 16):
        self.store, self.name, self.block = store, name, block_bytes
        self.on_corrupt = on_corrupt          # "raise" or "skip"
        self.skipped = 0
        size = size or store.size(name)       # the manifest knows the size: no HEAD request needed
        tail_off = max(0, size - tail_bytes)  # one read usually brings both the footer and the index
        tail = store.read(name, tail_off, size - tail_off)
        index_off, count, crc = parse_footer(tail[-FOOTER.size:])
        if index_off >= tail_off:
            index = tail[index_off - tail_off:index_off - tail_off + 8 * count]
        else:
            index = store.read(name, index_off, 8 * count)
        self.offsets = parse_index(index, count, crc)
        self.ends = self.offsets[1:] + [index_off]
        self.count = count

    def __len__(self):
        return self.count

    def _payload(self, i, raw):
        try:
            return check_record(raw, f"{self.name}#{i}")
        except CorruptRecord:
            if self.on_corrupt == "skip":
                self.skipped += 1
                return None
            raise

    def get(self, i):
        """Random access: one range request for one record."""
        a, b = self.offsets[i], self.ends[i]
        return self._payload(i, self.store.read(self.name, a, b - a))

    def records(self, start=0):
        """Stream records start..end. One request covers at least block_bytes, so many small records
        cost one round trip; a record larger than the block gets its own request."""
        buf, buf_off = b"", 0
        for i in range(start, self.count):
            a, b = self.offsets[i], self.ends[i]
            if a < buf_off or b > buf_off + len(buf):
                end = min(self.ends[-1], max(b, a + self.block))
                buf, buf_off = self.store.read(self.name, a, end - a), a
            yield i, self._payload(i, buf[a - buf_off:b - buf_off])


__all__ = ["ShardReader", "CorruptRecord", "REC"]
