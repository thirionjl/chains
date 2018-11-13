from collections import defaultdict
from typing import Dict


class NameGenerator:

    def __init__(self, prefix=None, separator="/"):
        self.prefix = prefix
        self.separator: str = separator
        self._max_id_per_prefix: Dict[str, int] = defaultdict(int)

    def generate(self, category: str) -> str:
        if self.prefix is not None:
            path = self.separator.join([self.prefix, category])
        else:
            path = category

        seq = self._max_id_per_prefix[path]
        self._max_id_per_prefix[path] = seq + 1
        return f"{path}:{seq}"
