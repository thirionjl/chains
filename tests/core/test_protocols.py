from typing import Protocol, Callable


class DummyProto(Protocol):
    def do1(self, arg: str, *args: str) -> None:
        ...


def adapted(func: Callable) -> Callable[..., None]:
    # def wrap(self, *args: int) -> None:
    #     print("\nwrapped")
    #     func(self, *args)

    return func


class Repeater:
    def __init__(self, x: int):
        self.x = x

    @adapted
    def do1(self, s: str) -> None:
        print(f"Repeater = {s * self.x}")


def do(doer: DummyProto) -> None:
    doer.do1("hi")


def testme() -> None:
    do(Repeater(3))
