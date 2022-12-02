import numpy as np

from chains.utils import nd_typing


def test_is_tensor():
    assert nd_typing.is_ndarray_like([43, 23])
    assert nd_typing.is_ndarray_like((11, 22))
    assert nd_typing.is_ndarray_like(-89.8)
    assert nd_typing.is_ndarray_like(np.arange(10).reshape(5, -1))

    assert not nd_typing.is_ndarray_like("aaa")
    assert not nd_typing.is_ndarray_like({"a": 56})
