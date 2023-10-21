# -*- coding: utf-8 -*-
"""
Backprop as the method of Lagrange multiplers (and even the implicit function
theorem).
"""
from __future__ import division
import numpy as np
from arsenal.alphabet import Alphabet
from arsenal.math.checkgrad import finite_difference


# Implementation choice: I decided to separate the input-copy and intermediate
# constraints to avoid annoyances with having two namespaces (x and z). I
# suppose writing all constraints as functions of x and z seems more general,
# but with input-copy consraints we don't any expressivity we just have handle
# them with special cases (easy enough).

class Computation:

    def __init__(self, f, inputs, constraints, df):
        self.d = len(inputs)
        self.n = self.d + len(constraints)
        self.constraints = constraints
        self.inputs = inputs
        self.f = f
        self.df = df

    def forward(self, x):
        "Evaluate `f(x)`"
        return self.f(self.solve_constraints(x))

    def solve_constraints(self, x):
        "Find a feasible solution to the constraints given `x`."
        z = np.zeros(self.n)
        z[self.inputs] = x
        for c in self.constraints:
            c.solve(z)
        return z

    def lagrangian(self, x, z, l):
        return (self.f(z)
                + l[:self.d].dot(x[:self.d] - z[:self.d])
                + l[self.d:].dot(self.constraints.penalties(z)))

    def dlagrangian(self, x, z, l):
        "Compute gradients of the Lagrangian wrt each argument."
        dx = np.zeros_like(x)
        dx += l[:self.d]

        dz = self.df(z) + self.dconstraints(z).dot(l)

        dl = np.zeros_like(l)
        dl[:self.d] += x[self.inputs] - z[self.inputs]
        dl[self.d:] += self.constraints.penalties(z)

        return dx, dz, dl

    def dconstraints(self, z):
        "Evaluate constraint matrix for `z`."
        # Note: The linear system approach build a massive highly structure
        # sparse matrix that represents the local gradients. This is really
        # inefficient in most cases because we can aggregate gradients as we
        # go. This avoids the need to materialize this monster matrix.
        D = np.zeros((self.n, self.n))
        D[self.inputs, self.inputs] = -1
        for c in self.constraints:
            c.grad(z, D[:, c.i])
        return D

    def reverse_mode(self, D, v):
        "Solve upper triangular linear system, `D x = -v`."
        l = v.copy()
        for c in reversed(self.constraints):
            for a in c.args:
                l[a] += l[c.i] * D[a, c.i]
        return l

    def forward_mode(self, D, v):
        "Solve upper triangular linear system, `Dᵀ = -v`."
        l = v.copy()
        for c in self.constraints:
            for a in c.args:
                l[c.i] += l[a] * D[a, c.i]
        return l


class Constraint:
    def __init__(self, i, f, args, df=None):
        self.args = args
        self.i = i
        self.f = f
        self.df = df
        if df is None:
            # Use finite-difference approximation if user didn't pass in df.
            self.df = lambda x: finite_difference(f)(x).flatten()

    def solve(self, z):
        # Closed form solution to the constraint, could take a gradient step or
        # solve a block-coordinate subproblem, more generally.
        z[self.i] = self.f(z[self.args])

    def penalty(self, z):
        return float(self.f(z[self.args]) - z[self.i])

    def grad(self, z, dz, adj=1):
        # Note: adjoint is scalar because constraint is scalar.
        dz[self.args] += adj * self.df(z[self.args])
        dz[self.i] += -adj


class Constraints(list):
    """Handles string-valued names and certain conventions like inputs need to be
    the first vars."""

    def __init__(self, inputs):
        self.A = Alphabet()
        self.inputs = self.A.map(inputs)  # need inputs to be the first vars
        super(Constraints, self).__init__()

    def add_constraint(self, lhs, f, rhs, df=None):
        self.append(Constraint(self.A[lhs], f, self.A.map(rhs), df))

    def penalties(self, z):
        return np.array([c.penalty(z) for c in self])
tests.py
# -*- coding: utf-8 -*-
"""
Backprop as the method of Lagrange multiplers (and even the implicit function
theorem).
"""
from __future__ import division
import numpy as np
import scipy.linalg
from lagrangeprop import Computation, Constraints
from arsenal.math.checkgrad import finite_difference, fdcheck
from arsenal.math import onehot, compare
from arsenal import colors


def test_implicit_diff_view(L):
    """
    Test connections between Lagrangian and implicit differentiation
    If you have the Lagrangian view of backprop, then implicit functions should
    really pop out!
    Think of forward propagation as a smooth blackbox function h that maps inputs
    (x) to intermediates (z).
      maximize f(z)
      subjecto h(x) = z
    Rewriting slightly, let g(x,z) = h(x) - z.
      maximize f(z)
      subjecto g(x,z) = 0
    With forward propagation we always satisfy the constraints, so g(x,z)=0. Thus,
    we also have "equilibrium" under little perturbations
      g(x+Δx, z+Δz) = g(x,z) + Δx ⋅ ∂g/∂x + Δz ⋅ ∂g/∂z = 0.
    Since g(x,z) = 0,
      Δx ⋅ ∂g/∂x + Δz ⋅ ∂g/∂z = 0
    Solve for Δz/Δx,
       Δz/Δx = - (∂g/∂z)^-1 ∂g/∂x  ← there's your linear system of equations!
    Combine with the objective ∂f/∂z
      ∂f/∂z Δz/Δx = ∂f/∂x
    """

    print colors.magenta % 'Implicit differentiation:',

    x = np.random.randn(L.d)

    # Important! connection only holds when constraints are satisfied!
    z = L.solve_constraints(x)

    f_dz_dx = finite_difference(L.solve_constraints)(x)

    dC_dx = np.zeros((L.n, L.d))
    dC_dx[L.inputs,L.inputs] = 1

    dC_dz = L.dconstraints(z)
    dz_dx = -scipy.linalg.solve(dC_dz.T, dC_dx).T

    assert np.allclose(f_dz_dx, dz_dx)

    df_dz = L.df(z)
    f_df_dx = finite_difference(L.forward)(x)

    assert np.allclose(f_df_dx, dz_dx.dot(df_dz))

    print colors.green % 'ok'


def test_forward_mode(L):
    print colors.magenta % 'Forward-mode:',

    x = np.random.randn(L.d)
    z = L.solve_constraints(x)
    D = L.dconstraints(z)

    # Compare methods to finite-difference approximation to ∇f(x)
    f_df_dx = finite_difference(L.forward)(x)

    # In forward mode, λ is interpreted as a vector of "tangents" pertaining to
    # a single input, instead of "adjoints" of the single output. Tangents are
    # equal to rows(cols?) of the Jacobian of the constraints.
    f_dz_dx = finite_difference(L.solve_constraints)(x)

    for i in range(L.d):   # loop over each input
        l = L.forward_mode(D, onehot(i, L.n))
        assert np.allclose(f_dz_dx[i], l)

        # df/dz * dz/dx[i] => df/dx[i]
        gi = L.df(z).dot(l)
        assert np.allclose(f_df_dx[i], gi)

    print colors.green % 'ok'


def test_dlagrangian(L):
    print colors.magenta % 'Finite-difference Lagrangian:',
    x = np.random.randn(L.d)
    z = np.random.uniform(-1,1,size=L.n)
    l = np.random.uniform(-1,1,size=L.n)

    dx, dz, dl = L.dlagrangian(x, z, l)
    assert fdcheck(lambda: L.lagrangian(x, z, l), z, dz, quiet=1).mean_relative_error < 0.01
    assert fdcheck(lambda: L.lagrangian(x, z, l), x, dx, quiet=1).mean_relative_error < 0.01
    assert fdcheck(lambda: L.lagrangian(x, z, l), l, dl, quiet=1).mean_relative_error < 0.01
    print colors.green % 'ok'


def test_reverse_mode(L):
    print colors.magenta % 'Reverse-mode:',
    x = np.random.randn(L.d)

    # Compare methods to finite-difference approximation to ∇f(x)
    f_df_dx = finite_difference(L.forward)(x)

    # run forward to cache all the relavant stuff.
    z = L.solve_constraints(x)
    l = L.reverse_mode(L.dconstraints(z), L.df(z))

    assert np.allclose(f_df_dx, l[:L.d])
    print colors.green % 'ok'


def test_linear_system(L):
    print colors.magenta % 'Linear solve:',
    x = np.random.randn(L.d)
    f_df_dx = finite_difference(L.forward)(x)

    z = L.solve_constraints(x)
    D = L.dconstraints(z)
    l = L.reverse_mode(D, L.df(z))

    # Run linear system solver -- Note that `linalg.solve` is generally worse at
    # solving the equations than `linalg.solve_triangular` (or equivalently
    # reverse mode). This is because the solver doesn't realize that the system
    # is upper triangular so it uses unstable operations like division and
    # subtraction.
    sol = scipy.linalg.solve(D, -L.df(z))
    assert np.allclose(l, sol)
    assert np.allclose(f_df_dx, sol[:L.d])

    # test aupper triangular solver
    sol = scipy.linalg.solve_triangular(D, -L.df(z))
    assert np.allclose(f_df_dx, sol[:L.d])
    assert np.allclose(l, sol)

    print colors.green % 'ok'


def test_blockcoordinate(L):
    print colors.magenta % 'Block-coordinate updates for z and λ:',

    x = np.random.randn(L.d)
    z = L.solve_constraints(x)
    l = L.reverse_mode(L.dconstraints(z), L.df(z))

    dx, dz, dl = L.dlagrangian(x, z, l)
    assert np.allclose(dx, l[:L.d])
    assert np.abs(dz).max() <= 1e-5
    assert np.allclose(dl, 0)

    print colors.green % 'ok'


def main():

    C = Constraints(['x','y'])
    C.add_constraint('a', np.exp,         ['x'],         df=np.exp)
    C.add_constraint('b', lambda x: x**2, ['a'],         df=lambda x: 2*x)
    C.add_constraint('c', np.sum,         ['a','b','y'], df=np.ones_like)
#    C.add_constraint('c', np.product,     ['a','b','y'])
#    C.add_constraint('d', np.exp,         ['c'],         df=np.exp)
    C.add_constraint('d', np.tanh,        ['c'])
    C.add_constraint('e', np.sin,         ['c'],         df=np.cos)
    C.add_constraint('f', np.sum,         ['d','e'],     df=np.ones_like)

    n = len(C.inputs) + len(C)
    _r = np.random.randn(n)    # random linear function of intermediate nodes
    f = _r.dot
    df = lambda z: _r.copy()

    L = Computation(f, C.inputs, C, df = df)

    test_dlagrangian(L)
    test_reverse_mode(L)
    test_forward_mode(L)
    test_linear_system(L)
    test_blockcoordinate(L)
    test_implicit_diff_view(L)


if __name__ == '__main__':
    main()
