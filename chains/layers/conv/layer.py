from . import filter_utils as fu
from ..fc import layer as fcl

# TODO Layer factories
def forward(X, F, b, stride=1):
    XC = fu.im2col_activations(X, F, stride)
    FC = fu.im2col_filters(X, F)
    ZC = fcl.forward(XC, FC, b)
    return fu.reshape_out(ZC, X, F, stride), XC, FC, ZC


def backward(dZ, X, XC, F, FC, b, stride=1):
    m, c, nh, nw, d, fh, fw = fu.all_dimensions(X, F)
    mm, dd, cnt_h, cnt_w = dZ.shape
    if mm != m:
        raise ValueError(f"Number of examples im activations({m}) is different from number of examples \
        in output derivatives({mm})")

    if dd != d:
        raise ValueError(f"Number of filters in input({d}) is different from number of filters \
        in output derivatives({dd})")

    dZC = dZ.transpose(0, 1, 2, 3).reshape(d, m * cnt_h * cnt_w)
    (dXC, dFC, db) = fcl.backward(dZC, XC, FC, b)
    dF = dFC.reshape(c, c, fh, fw)
    dX = fu.im2col_activations_back(dXC, X, F, stride)
    return dX, dF, db
