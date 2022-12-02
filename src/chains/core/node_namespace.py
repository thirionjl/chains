from collections import deque
from contextvars import ContextVar

NODE_CONTEXT: ContextVar[tuple[str, ...]] = ContextVar("node_group")


def _current() -> tuple[str, ...]:
    try:
        return NODE_CONTEXT.get()
    except LookupError:
        return tuple()


def current_ns() -> str:
    return ".".join(_current())


class NodeNamespace:
    def __init__(self, ctx: str):
        self.ctx: str = ctx

    def __enter__(self):
        if self.ctx:
            new_ctx = _current() + (self.ctx,)
            self.old_ctx_token = NODE_CONTEXT.set(new_ctx)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx:
            NODE_CONTEXT.reset(self.old_ctx_token)
