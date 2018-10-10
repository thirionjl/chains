from chains.tensor.tensor import Shape, Dim


def test_is_broadcast_compatible():
    assert Shape(2, 5).is_broadcast_compatible(Shape(2, 5))
    assert Shape(2, 5).is_broadcast_compatible(Shape(1))
    assert Shape(3, 1).is_broadcast_compatible(Shape(1, 3))
    assert Shape(7, 3, 3).is_broadcast_compatible(Shape(3, 3))
    assert Shape(None, 3, 3).is_broadcast_compatible(Shape(3, 1))
    assert Shape(None, 4).is_broadcast_compatible(Shape())
    assert Shape(None, None).is_broadcast_compatible(Shape(1))
    assert Shape(None, 1).is_broadcast_compatible(Shape(1, None))

    assert not Shape(2, 5).is_broadcast_compatible(Shape(3))
    assert not Shape(None, 1).is_broadcast_compatible(Shape(None, 1))
    assert not Shape(1, None).is_broadcast_compatible(Shape(None))


def test_broadcasted_shape():
    assert Shape(2, 5).broadcast(Shape(2, 5)) == Shape(2, 5)
    assert Shape(2, 5).broadcast(Shape(1)) == Shape(2, 5)
    assert Shape(3, 1).broadcast(Shape(1, 3)) == Shape(3, 3)
    assert Shape(7, 3, 3).broadcast(Shape(3, 3)) == Shape(7, 3, 3)

    m = Dim.unknown()
    n = Dim.unknown()
    assert Shape(m, 3, 3).broadcast(Shape(3, 1)) == Shape(m, 3, 3)
    assert Shape(m, 4).broadcast(Shape()) == Shape(m, 4)
    assert Shape(m, n).broadcast(Shape(1)) == Shape(m, n)
    assert Shape(m, 1).broadcast(Shape(1, n)) == Shape(m, n)


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

    assert Shape(2, 3).is_assignable_to(Shape(2, 3))
    assert Shape(2, 12).is_assignable_to(Shape(2, m))
    assert Shape(2, m).is_assignable_to(Shape(2, m))
    assert Shape(2, m).is_assignable_to(Shape(n, m))

    assert not Shape(2).is_assignable_to(Shape(2, m))
    assert not Shape(2).is_assignable_to(Shape(3))
    assert not Shape(2, m).is_assignable_to(Shape(2, n))
