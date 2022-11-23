from collections import defaultdict
from typing import Optional, Mapping


class NameGenerator:
    def __init__(self, prefix: Optional[str] = None, separator: str = "/"):
        self.prefix = prefix
        self.separator = separator
        self._max_id_per_prefix: dict[str, int] = defaultdict(int)

    def generate(self, category: str) -> str:
        if self.prefix is not None:
            path = self.separator.join([self.prefix, category])
        else:
            path = category

        seq = self._max_id_per_prefix[path]
        self._max_id_per_prefix[path] = seq + 1
        return f"{path}:{seq}"
