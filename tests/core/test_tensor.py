import numpy as np

from chains.core import tensor


def test_is_tensor():
    assert tensor.is_tensor([43, 23])
    assert tensor.is_tensor((11, 22))
    assert tensor.is_tensor(-89.8)
    assert tensor.is_tensor(np.arange(10).reshape(5, -1))

    assert not tensor.is_tensor("aaa")
    assert not tensor.is_tensor({"a": 56})
