"""TCP links for a ring of workers: rank r sends to rank r+1 and receives from rank r-1."""
import queue
import socket
import struct
import threading
import time

import numpy as np

HEADER = struct.Struct("!IQ")          # tag (uint32), payload length in bytes (uint64)


class CommError(RuntimeError):
    """Base class: the collective cannot finish."""


class CommTimeout(CommError):
    """A neighbour sent nothing for `timeout` seconds (hung or very slow peer)."""


class PeerClosed(CommError):
    """A neighbour closed its socket (it crashed or gave up)."""


def listen(host="127.0.0.1", port=0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(4)
    return s


class RingLink:
    def __init__(self, rank, world, addrs, timeout=30.0, listen_sock=None):
        self.rank, self.world, self.timeout = rank, world, timeout
        self.left, self.right = (rank - 1) % world, (rank + 1) % world
        self.bytes_sent = self.bytes_recv = 0          # payload bytes (what the theory counts)
        self.wire_sent = 0                             # payload + headers
        self._err = None
        if world == 1:
            return
        srv = listen_sock or listen("0.0.0.0", addrs[rank][1])   # multi-host: listen on all interfaces
        self.tx = self._connect(addrs[self.right])     # our outgoing edge of the ring
        srv.settimeout(timeout)
        try:
            self.rx, _ = srv.accept()                  # incoming edge, from the left neighbour
        except socket.timeout:
            raise CommTimeout(f"rank {rank}: rank {self.left} never connected") from None
        srv.close()
        who = struct.unpack("!I", self._read_exact(4))[0]
        if who != self.left:
            raise CommError(f"rank {rank}: expected rank {self.left}, got {who}")
        for s in (self.tx, self.rx):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(timeout)
        self._q = queue.Queue()
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()

    def _connect(self, addr):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                s = socket.create_connection(tuple(addr), timeout=self.timeout)
                s.sendall(struct.pack("!I", self.rank))
                return s
            except OSError:
                if time.monotonic() > deadline:
                    raise CommTimeout(f"rank {self.rank}: cannot reach rank {self.right} at {addr}") from None
                time.sleep(0.05)

    # ---- sending: a background thread, so that send and receive happen at the same time
    def send(self, arr, tag):
        if self._err:
            raise self._err
        data = np.ascontiguousarray(arr).tobytes()     # a copy: the caller may change arr right away
        self._q.put(HEADER.pack(tag, len(data)) + data)
        self.bytes_sent += len(data)
        self.wire_sent += HEADER.size + len(data)

    def _send_loop(self):
        while True:
            msg = self._q.get()
            if msg is None:
                return
            try:
                self.tx.sendall(msg)
            except socket.timeout:
                self._err = CommTimeout(f"rank {self.rank}: rank {self.right} stopped reading")
                return
            except OSError as e:
                self._err = PeerClosed(f"rank {self.rank}: rank {self.right} is gone ({e.__class__.__name__})")
                return

    # ---- receiving
    def _read_exact(self, n, out=None):
        buf = out if out is not None else bytearray(n)
        view, got = memoryview(buf).cast("B"), 0
        while got < n:
            try:
                k = self.rx.recv_into(view[got:], n - got)
            except socket.timeout:
                raise CommTimeout(f"rank {self.rank}: nothing from rank {self.left} for {self.timeout:g} s") from None
            except OSError as e:
                raise PeerClosed(f"rank {self.rank}: rank {self.left} is gone ({e.__class__.__name__})") from None
            if k == 0:
                raise PeerClosed(f"rank {self.rank}: rank {self.left} closed the connection")
            got += k
        return buf

    def recv(self, n_items, tag, dtype=np.float64):
        tag_in, nbytes = HEADER.unpack(self._read_exact(HEADER.size))
        if tag_in != tag or nbytes != n_items * np.dtype(dtype).itemsize:
            raise CommError(f"rank {self.rank}: out of step (tag {tag_in} != {tag} or {nbytes} bytes)")
        out = np.empty(n_items, dtype)
        self._read_exact(nbytes, out)
        self.bytes_recv += nbytes
        return out

    def close(self):
        if self.world == 1:
            return
        self._q.put(None)
        self._sender.join(timeout=1.0)
        for s in (self.tx, self.rx):
            try:
                s.close()
            except OSError:
                pass
