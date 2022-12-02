from chains.core.shape import Shape, Dim


def test_is_broadcast_compatible():
    assert Shape.of(2, 5).is_broadcast_compatible(Shape.of(2, 5))
    assert Shape.of(2, 5).is_broadcast_compatible(Shape.of(1))
    assert Shape.of(3, 1).is_broadcast_compatible(Shape.of(1, 3))
    assert Shape.of(7, 3, 3).is_broadcast_compatible(Shape.of(3, 3))
    assert Shape.of(None, 3, 3).is_broadcast_compatible(Shape.of(3, 1))
    assert Shape.of(None, 4).is_broadcast_compatible(Shape.of())
    assert Shape.of(None, None).is_broadcast_compatible(Shape.of(1))
    assert Shape.of(None, 1).is_broadcast_compatible(Shape.of(1, None))

    assert not Shape.of(2, 5).is_broadcast_compatible(Shape.of(3))
    assert not Shape.of(None, 1).is_broadcast_compatible(Shape.of(None, 1))
    assert not Shape.of(1, None).is_broadcast_compatible(Shape.of(None))


def test_broadcasted_shape():
    x = Shape.of(2, 5).broadcast(Shape.of(2, 5))
    y = Shape.of(2, 5)
    assert x == y
    assert Shape.of(2, 5).broadcast(Shape.of(1)) == y
    assert Shape.of(3, 1).broadcast(Shape.of(1, 3)) == Shape.of(3, 3)
    assert Shape.of(7, 3, 3).broadcast(Shape.of(3, 3)) == Shape.of(7, 3, 3)

    m = Dim.unknown()
    n = Dim.unknown()
    assert Shape.of(m, 3, 3).broadcast(Shape.of(3, 1)) == Shape.of(m, 3, 3)
    assert Shape.of(m, 4).broadcast(Shape.of()) == Shape.of(m, 4)
    assert Shape.of(m, n).broadcast(Shape.of(1)) == Shape.of(m, n)
    assert Shape.of(m, 1).broadcast(Shape.of(1, n)) == Shape.of(m, n)


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

    assert Shape.of(2, 3).is_assignable_to(Shape.of(2, 3))
    assert Shape.of(2, 12).is_assignable_to(Shape.of(2, m))
    assert Shape.of(2, m).is_assignable_to(Shape.of(2, m))
    assert Shape.of(2, m).is_assignable_to(Shape.of(n, m))

    assert not Shape.of(2).is_assignable_to(Shape.of(2, m))
    assert not Shape.of(2).is_assignable_to(Shape.of(3))
    assert not Shape.of(2, m).is_assignable_to(Shape.of(2, n))


def test_reduce_along_axis():
    m = Dim.unknown()
    shape = Shape.of(2, 3, m, 5)

    assert shape.reduce_along_axis(axis=0) == Shape.of(3, m, 5)
    assert shape.reduce_along_axis(axis=-1) == Shape.of(2, 3, m)
    assert shape.reduce_along_axis(axis=[0, 1]) == Shape.of(m, 5)
    assert shape.reduce_along_axis(axis=[-1, -1, 1]) == Shape.of(2, m)
    assert shape.reduce_along_axis(axis=[-1, -2]) == Shape.of(2, 3)

    assert shape.reduce_along_axis(axis=0, keep_dims=True) == Shape.of(1, 3, m, 5)
    assert shape.reduce_along_axis(axis=-1, keep_dims=True) == Shape.of(2, 3, m, 1)
    assert shape.reduce_along_axis(axis=[0, 1], keep_dims=True) == Shape.of(1, 1, m, 5)
    assert shape.reduce_along_axis(axis=[-1, -1, 1], keep_dims=True) == Shape.of(
        2, 1, m, 1
    )
    assert shape.reduce_along_axis(axis=[-1, -2], keep_dims=True) == Shape.of(
        2, 3, 1, 1
    )


def test_transpose():
    m = Dim.unknown()
    shape = Shape.of(2, 3, m)

    assert shape.transpose((1, 0, 2)) == Shape.of(3, 2, m)
    assert shape.transpose(1, 0, 2) == Shape.of(3, 2, m)
    assert shape.transpose(2, 1, 0) == Shape.of(m, 3, 2)
