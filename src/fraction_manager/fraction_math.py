from fractions import Fraction

import logging 

_logger = logging.getLogger(__name__)

def gcd(a: int, b: int):
    if b == 0:
        return a
    return gcd(b, a % b)

def trace_poly(p: int, q: int, ta: complex, tB: complex, taB: complex):
    #See pp. 286
    #Can't use Fraction as we use improper one (1/0) 
    fract = (p, q)
    if fract == (0, 1):
        return ta
    elif fract == (1, 0):
        return tB

    tr_u = ta
    tr_v = tB
    tr_uv = taB

    f1 = (0, 1)
    f2 = (1, 0)
    f3 = (1, 1)


    while (f3 != fract):
        if (fract[0]/fract[1] < f3[0]/f3[1]):
            f2 = f3
            f3 = (f1[0] + f3[0], f1[1] + f3[1])

            temp = tr_uv

            tr_uv = tr_u * tr_uv - tr_v
            tr_v = temp
        else:
            f1 = f3
            f3 = (f2[0] + f3[0], f2[1] + f3[1])

            temp = tr_uv

            tr_uv = tr_v * tr_uv - tr_u
            tr_u = temp
    return tr_uv

def trace_equation(fract: Fraction, mu: complex):
    return trace_poly(fract.numerator, fract.denominator, -1j * mu, 2, -1j * mu + 2j) - 2
