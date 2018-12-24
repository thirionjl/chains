from chains.core.static_shape import StaticShape, Dim


def test_is_broadcast_compatible():
    assert StaticShape(2, 5).is_broadcast_compatible(StaticShape(2, 5))
    assert StaticShape(2, 5).is_broadcast_compatible(StaticShape(1))
    assert StaticShape(3, 1).is_broadcast_compatible(StaticShape(1, 3))
    assert StaticShape(7, 3, 3).is_broadcast_compatible(StaticShape(3, 3))
    assert StaticShape(None, 3, 3).is_broadcast_compatible(StaticShape(3, 1))
    assert StaticShape(None, 4).is_broadcast_compatible(StaticShape())
    assert StaticShape(None, None).is_broadcast_compatible(StaticShape(1))
    assert StaticShape(None, 1).is_broadcast_compatible(StaticShape(1, None))

    assert not StaticShape(2, 5).is_broadcast_compatible(StaticShape(3))
    assert not StaticShape(None, 1).is_broadcast_compatible(
        StaticShape(None, 1))
    assert not StaticShape(1, None).is_broadcast_compatible(StaticShape(None))


def test_broadcasted_shape():
    assert StaticShape(2, 5).broadcast(StaticShape(2, 5)) == StaticShape(2, 5)
    assert StaticShape(2, 5).broadcast(StaticShape(1)) == StaticShape(2, 5)
    assert StaticShape(3, 1).broadcast(StaticShape(1, 3)) == StaticShape(3, 3)
    assert StaticShape(7, 3, 3).broadcast(StaticShape(3, 3)) == StaticShape(7,
                                                                            3,
                                                                            3)

    m = Dim.unknown()
    n = Dim.unknown()
    assert StaticShape(m, 3, 3).broadcast(StaticShape(3, 1)) == StaticShape(m,
                                                                            3,
                                                                            3)
    assert StaticShape(m, 4).broadcast(StaticShape()) == StaticShape(m, 4)
    assert StaticShape(m, n).broadcast(StaticShape(1)) == StaticShape(m, n)
    assert StaticShape(m, 1).broadcast(StaticShape(1, n)) == StaticShape(m, n)


def test_dim_is_assignable_to():
    m = Dim.unknown()
    n = Dim.unknown()

    assert Dim(2).is_assignable_to(Dim(2))
    assert Dim(12).is_assignable_to(m)
    assert m.is_assignable_to(m)

    assert not Dim(2).is_assignable_to(Dim(3))
    assert not m.is_assignable_to(Dim(3))
    assert not m.is_assignable_to(n)


def test_shape_is_assignable_to():
    m = Dim.unknown()
    n = Dim.unknown()

    assert StaticShape(2, 3).is_assignable_to(StaticShape(2, 3))
    assert StaticShape(2, 12).is_assignable_to(StaticShape(2, m))
    assert StaticShape(2, m).is_assignable_to(StaticShape(2, m))
    assert StaticShape(2, m).is_assignable_to(StaticShape(n, m))

    assert not StaticShape(2).is_assignable_to(StaticShape(2, m))
    assert not StaticShape(2).is_assignable_to(StaticShape(3))
    assert not StaticShape(2, m).is_assignable_to(StaticShape(2, n))


def test_reduce_along_axis():
    m = Dim.unknown()
    shape = StaticShape(2, 3, m, 5)

    assert shape.reduce_along_axis(axis=0) == StaticShape(3, m, 5)
    assert shape.reduce_along_axis(axis=-1) == StaticShape(2, 3, m)
    assert shape.reduce_along_axis(axis=[0, 1]) == StaticShape(m, 5)
    assert shape.reduce_along_axis(axis=[-1, -1, 1]) == StaticShape(2, m)
    assert shape.reduce_along_axis(axis=[-1, -2]) == StaticShape(2, 3)

    assert shape.reduce_along_axis(axis=0,
                                   keep_dims=True) == StaticShape(1, 3, m, 5)
    assert shape.reduce_along_axis(axis=-1,
                                   keep_dims=True) == StaticShape(2, 3, m, 1)
    assert shape.reduce_along_axis(axis=[0, 1],
                                   keep_dims=True) == StaticShape(1, 1, m, 5)
    assert shape.reduce_along_axis(axis=[-1, -1, 1],
                                   keep_dims=True) == StaticShape(2, 1, m, 1)
    assert shape.reduce_along_axis(axis=[-1, -2],
                                   keep_dims=True) == StaticShape(2, 3, 1, 1)


def test_transpose():
    m = Dim.unknown()
    shape = StaticShape(2, 3, m)

    assert shape.transpose((1, 0, 2)) == StaticShape(3, 2, m)
    assert shape.transpose(1, 0, 2) == StaticShape(3, 2, m)
    assert shape.transpose(2, 1, 0) == StaticShape(m, 3, 2)


def test_flatten():
    m = Dim.unknown()

    assert StaticShape(2, 3, m).flatten_axis(keep_axis=-1) == StaticShape(6, m)
    assert StaticShape(2, 3, m).flatten_axis(keep_axis=2) == StaticShape(6, m)
    assert StaticShape(m, 2, 3).flatten_axis(keep_axis=0) == StaticShape(m, 6)
    assert StaticShape(m, 6).flatten_axis(keep_axis=-2) == StaticShape(m, 6)

    shape = StaticShape(2, 3, m).flatten_axis(keep_axis=0)
    assert len(shape) == 2
    assert shape[0] == Dim.of(2)
    assert shape[1].is_unknown()
