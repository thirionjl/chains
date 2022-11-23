from typing import Mapping, Any

from .ops import Op, Var
from .ops_norm import BatchNormPredict, BatchNormTraining
from .tensor import Tensor

registered_savers = []


def register(saver_class: type):
    # validate.is_a("saver_class", saver_class, Saver)
    registered_savers.append(saver_class())
    print(f"[DEBUG] Register saver class {saver_class}")


class MetaSaver(type):
    def __new__(meta, name, bases, class_dict) -> Any:
        cls = type.__new__(meta, name, bases, class_dict)
        register(cls)
        return cls


class Saver(metaclass=MetaSaver):
    @classmethod
    def accepts(cls, op: Op) -> bool:
        return False

    def save(self, op: Op) -> Mapping[str, Tensor]:
        raise NotImplementedError

    def restore(self, op: Op, values: Mapping[str, Tensor]) -> None:
        raise NotImplementedError


def save(op: Op):
    for saver in registered_savers:
        if saver.accepts(op):
            return saver.save(op)
    return None


def restore(op: Op, values: Mapping[str, Tensor]) -> None:
    for saver in registered_savers:
        if saver.accepts(op):
            saver.restore(op, values)


class VarSaver(Saver):
    @classmethod
    def accepts(cls, op: Op) -> bool:
        return isinstance(op, Var)

    def save(self, op: Op) -> Mapping[str, Tensor]:
        return {"value": op.output}

    def restore(self, op: Op, values: Mapping[str, Tensor]) -> None:
        op.output = values["value"]


class BatchNormSaver(Saver):
    @classmethod
    def accepts(cls, op: Op) -> bool:
        return isinstance(op, BatchNormPredict) or isinstance(op, BatchNormTraining)

    def save(self, op):
        return {"avg": op.avg, "var": op.var}

    def restore(self, op, values: Mapping[str, Tensor]):
        op.avg = values["avg"]
        op.var = values["var"]
