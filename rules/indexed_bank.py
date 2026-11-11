"""Indexed, relation-at-a-time loading of generated routed-rule TSV files.

The sidecar indexes byte ranges only. Weights are parsed from the original
decimal strings with Python float, preserving the eager loader's arithmetic.
Only unquoted numeric head IDs in the first column are supported.
"""
import csv
import io
import json
import mmap
import os
from pathlib import Path
import re
import time


def _signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


class IndexedRuleBank:
    def __init__(self, path, cache_path, weight_field="weight"):
        started = time.perf_counter()
        self.path = Path(path).resolve()
        self.weight_field = weight_field
        self.materialization_seconds = 0.0
        self.loaded_heads = 0
        self.signature = _signature(self.path)
        cache_path = Path(cache_path)
        cached = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
            except (ValueError, OSError):
                pass
        self.index_reused = bool(cached and cached.get("version") == 1
                                 and cached.get("signature") == self.signature
                                 and cached.get("source") == str(self.path))
        if not self.index_reused:
            cached = self._index()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(cached))
            os.replace(temporary, cache_path)
        self.columns = cached["columns"]
        for name in ("head_id", "body_ids", weight_field, "group_id"):
            if name not in self.columns:
                raise ValueError(f"Missing routed-rule column: {name}")
        self.heads = cached["heads"]
        self.index_seconds = time.perf_counter() - started

    def _index(self):
        heads = {}
        with self.path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            start = data.find(b"\n") + 1
            if not start:
                raise ValueError("Rule TSV has no header newline")
            columns = next(csv.reader([data[:start].decode().strip()], delimiter="\t"))
            if columns[0] != "head_id":
                raise ValueError("Indexed rules require head_id as the first column")
            scalar_id = 0
            while start < len(data):
                match = re.match(rb"([0-9]+)\t", data[start:start + 32])
                if match is None:
                    raise ValueError(f"Invalid numeric rule head at byte {start}")
                head_bytes = match[1]
                # Scan in native regex code to the next different relation.
                next_head = re.compile(rb"^(?!" + head_bytes + rb"\t)[0-9]+\t", re.MULTILINE).search(data, start)
                end = next_head.start() if next_head else len(data)
                block = data[start:end]
                count = block.count(b"\n") + int(not block.endswith(b"\n"))
                heads.setdefault(str(int(head_bytes)), []).append([start, end, scalar_id, count])
                scalar_id += count
                start = end
        if _signature(self.path) != self.signature:
            raise RuntimeError("Rule file changed while indexing")
        return dict(version=1, source=str(self.path), signature=self.signature,
                    columns=columns, heads=heads, scalar_instances=scalar_id)

    def get(self, head, default=None):
        blocks = self.heads.get(str(head))
        return IndexedHeadRules(self, int(head), blocks) if blocks else default


class IndexedHeadRules:
    def __init__(self, bank, head, blocks):
        self.bank, self.head, self.blocks = bank, head, blocks
        self._compact = None

    def __len__(self):
        return sum(b[3] for b in self.blocks)

    def compact(self):
        if self._compact is not None:
            return self._compact
        started = time.perf_counter()
        bank = self.bank
        if _signature(bank.path) != bank.signature:
            raise RuntimeError("Rule file changed after indexing")
        c = {name: i for i, name in enumerate(bank.columns)}
        compact = {}
        width = 0
        with bank.path.open("rb") as stream:
            for start, end, first_id, count in self.blocks:
                stream.seek(start)
                rows = csv.reader(io.StringIO(stream.read(end - start).decode()), delimiter="\t")
                seen = 0
                for offset, row in enumerate(rows):
                    if int(row[c["head_id"]]) != self.head:
                        raise ValueError("Rule index relation mismatch")
                    body = tuple(map(int, row[c["body_ids"]].split()))
                    group = int(row[c["group_id"]])
                    if group < 0:
                        raise ValueError("Negative rule group")
                    weight = float(row[c[bank.weight_field]])
                    width = max(width, group + 1)
                    if body not in compact:
                        compact[body] = [first_id + offset, {}]
                    item = compact[body]
                    # Match the original compactor's duplicate semantics.
                    if item[1].get(group, 0.0) != 0.0:
                        raise ValueError("duplicate Horn rule/group instance")
                    item[1][group] = weight
                    seen += 1
                if seen != count:
                    raise ValueError("Rule index row-count mismatch")
        self._compact = [dict(rule_id=first_id, head=self.head, body=body,
                              weight=1.0, group_id=None,
                              rank_weights=tuple(weights.get(i, 0.0) for i in range(width)))
                         for body, (first_id, weights) in compact.items()]
        bank.materialization_seconds += time.perf_counter() - started
        bank.loaded_heads += 1
        return self._compact
