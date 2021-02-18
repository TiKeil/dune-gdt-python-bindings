#!/usr/bin/env python
# coding: utf-8
# %%

# # Tutorial 20 [WIP]: discontinuous IPDG for the stationary heat equation
# 
# This tutorial shows how to solve the stationary heat equation with homogeneous Dirichlet boundary conditions using interior penalty (IP) discontinuous Galerkin (DG) Finite Elmenets with `dune-gdt`.
# 
# ## This is work in progress (WIP), still missing:
# 
# * mathematical theory on IPDG methods
# * explanation of the IPDG implementation
# * non-homonegenous Dirichlet boundary values
# * Neumann boundary values
# * Robin boundary values

# %%
import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings


# %%


from dune.xt.grid import Dim
from dune.xt.functions import ConstantFunction, ExpressionFunction

d = 2
omega = ([0, 0], [1, 1])

kappa = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='kappa')
# note that we need to prescribe the approximation order, which determines the quadrature on each element
f = ExpressionFunction(dim_domain=Dim(d), variable='x', expression='exp(x[0]*x[1])', order=3, name='f')


# %%


from dune.xt.grid import Simplex, make_cube_grid, visualize_grid

grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])
grid.global_refine(1) # we need to refine once to obtain a symmetric grid

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')

# # 1.9: everything in a single function
# 
# For a better overview, the above discretization code is also available in a single function in the file `discretize_elliptic_ipdg.py`.

# %%


from numbers import Number

from dune.xt.grid import (
    AllDirichletBoundaryInfo,
    ApplyOnCustomBoundaryIntersections,
    ApplyOnInnerIntersectionsOnce,
    ApplyOnInnerIntersections,
    Dim,
    DirichletBoundary,
    Walker,
)
from dune.xt.functions import GridFunction as GF


# %%


from dune.gdt import (
    BilinearForm,
    DiscontinuousLagrangeSpace,
    DiscreteFunction,
    LocalElementIntegralBilinearForm,
    LocalElementIntegralFunctional,
    LocalElementProductIntegrand,
    LocalCouplingIntersectionIntegralBilinearForm,
    LocalIPDGBoundaryPenaltyIntegrand,
    LocalIPDGInnerPenaltyIntegrand,
    LocalIntersectionIntegralBilinearForm,
    LocalLaplaceIntegrand,
    LocalLaplaceIPDGDirichletCouplingIntegrand,
    LocalLaplaceIPDGInnerCouplingIntegrand,
    MatrixOperator,
    VectorFunctional,
    estimate_combined_inverse_trace_inequality_constant,
    estimate_element_to_intersection_equivalence_constant,
    make_element_and_intersection_sparsity_pattern,
)


# %%


print(grid)
ApplyOnInnerIntersectionsOnce(grid)


# %%


d = grid.dimension
diffusion = GF(grid, kappa, dim_range=(Dim(d), Dim(d)))
source = GF(grid, f)
weight = GF(grid, 1, dim_range=(Dim(d), Dim(d)))

boundary_info = AllDirichletBoundaryInfo(grid)
penalty_parameter = 16
symmetry_factor = 1
V_h = DiscontinuousLagrangeSpace(grid, order=1)
if not penalty_parameter:
    # TODO: check if we need to include diffusion for the coercivity here!
    # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
    C_G = estimate_element_to_intersection_equivalence_constant(grid)
    C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
    penalty_parameter = C_G*C_M_times_1_plus_C_T
    if symmetry_factor == 1:
        penalty_parameter *= 4
assert isinstance(penalty_parameter, Number)
assert penalty_parameter > 0

l_h = VectorFunctional(grid, source_space=V_h)
l_h += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(source))

a_h = MatrixOperator(grid, source_space=V_h, range_space=V_h,
                     sparsity_pattern=make_element_and_intersection_sparsity_pattern(V_h))
a_form = BilinearForm(grid)
a_form += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(diffusion))
a_form += (LocalCouplingIntersectionIntegralBilinearForm(
                LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
                + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)),
            ApplyOnInnerIntersectionsOnce(grid))
# a_form += LocalCouplingIntersectionIntegralBilinearForm(
#                 LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
#                 + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight))
a_form += (LocalIntersectionIntegralBilinearForm(
                LocalIPDGBoundaryPenaltyIntegrand(penalty_parameter, weight)
                + LocalLaplaceIPDGDirichletCouplingIntegrand(symmetry_factor, diffusion)),
            ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
a_h.append(a_form)


# %%


walker = Walker(grid)
walker.append(a_h)
walker.append(l_h)


# %%


walker.walk()


# %%


u_h = DiscreteFunction(V_h, name='u_h')
a_h.apply_inverse(l_h.vector, u_h.dofs.vector)

from dune.gdt import visualize_function
_ = visualize_function(u_h)


# %%


from dune.gdt import visualize_function

u_h = discretize_elliptic_ipdg_dirichlet_zero(
    grid, kappa, f,
    symmetry_factor=1, penalty_parameter=16, weight=1) # SIPDG scheme

_ = visualize_function(u_h)

