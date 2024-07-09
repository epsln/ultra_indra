from fractions import Fraction
from scipy.optimize import root_scalar

def trace_poly(fract: Fraction, ta: complex, tB: complex, taB: complex):
    #See pp. 286
    if fract == Fraction(0, 1):
        return ta
    elif fract == Fraction(1, 0):
        return tB

    tr_u = ta
    tr_v = tB
    tr_uv = taB

    while (f3 != fract):
        if (fract < f3):
            f2 = f3
            f3 = Fraction(f1.numerator + f3.numerator, f1.denominator + f3.denominator)

            temp = tr_uv

            tr_uv = tr_u * tr_uv - tr_v
            tr_v = temp
        else:
            f1 = f3
            f3 = Fraction(f2.numerator + f3.numerator, f2.denominator + f3.denominator)

            temp = tr_uv

            tr_uv = tr_v * tr_uv - tr_u
            tr_u = temp

    return tr_uv

def trace_equation(fract: Fraction, mu: complex):
    return trace_poly(fract, -1j * mu, 2, -1j * mu + 2j) - 2

def trace_solver(fract: Fraction):
    return root_scalar(self, trace_equation, method = 'brent', xtol = self.root_epsilon, maxiter = self.root_maxiter)
