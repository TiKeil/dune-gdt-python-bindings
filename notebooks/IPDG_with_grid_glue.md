---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# wurlitzer: display dune's output in the notebook
%load_ext wurlitzer
%matplotlib notebook

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings
```

## 1. Creating a DD grid

Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid
from dune.xt.grid import AllDirichletBoundaryInfo
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
macro_grid = make_cube_grid(Dim(d), Simplex(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])
macro_grid.global_refine(1)

macro_boundary_info = AllDirichletBoundaryInfo(macro_grid)

print(f'grid has {macro_grid.size(0)} elements, {macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')
```

Now we can use this grid as a macro grid for a dd grid.

```python
# start with no refinement on the subdomains
dd_grid = make_cube_dd_grid(macro_grid, 2)
```

```python
from dune.xt.grid import visualize_grid
_ = visualize_grid(macro_grid)
```

# 2. Creating micro CG spaces


We can define cg spaces for every local grid

```python
from dune.gdt import ContinuousLagrangeSpace

S = dd_grid.num_subdomains
spaces = [ContinuousLagrangeSpace(dd_grid.local_grid(ss), order=1) for ss in range(S)]
grids = [dd_grid.local_grid(ss) for ss in range(S)]
neighbors = [dd_grid.neighbors(ss) for ss in range(S)]
```

# 3. Creating a BlockOperator for pymor

```python
from dune.xt.grid import Dim
from dune.xt.functions import ConstantFunction, ExpressionFunction
from dune.xt.functions import GridFunction

d = 2
omega = ([0, 0], [1, 1])

kappa = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[1.], name='kappa')
f = ExpressionFunction(dim_domain=Dim(d), variable='x', expression='exp(x[0]*x[1])', order=3, name='f')
```

```python
from dune.gdt import (BilinearForm,
                      MatrixOperator,
                      make_element_sparsity_pattern,
                      make_element_and_intersection_sparsity_pattern,
                      LocalLaplaceIntegrand,
                      LocalElementIntegralBilinearForm,
                      DirichletConstraints)
from dune.xt.grid import Walker


def assemble_local_op(grid, space, boundary_info, d):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(
        LocalLaplaceIntegrand(GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))))
    a_h.append(a_form)
    
    dirichlet_constraints = DirichletConstraints(boundary_info, space)
    
    #walker on local grid
    walker = Walker(grid)
    walker.append(a_h)
    walker.append(dirichlet_constraints)
    walker.walk()
    
#     print('centers: ', grid.centers())
#     print(dirichlet_constraints.dirichlet_DoFs)
    print(a_h.matrix)
    a_h.assemble()
    print(a_h.matrix.__repr__())
    return a_h
```

```python
ops = np.empty((S, S), dtype=object)
```

```python
for ss in range(S):
    space = spaces[ss]
    grid = dd_grid.local_grid(ss)
    boundary_info = dd_grid.macro_based_boundary_info(ss, macro_boundary_info)
    ops[ss, ss] = assemble_local_op(grid, space, boundary_info, d)
```

```python
from dune.gdt import LocalCouplingIntersectionIntegralBilinearForm, LocalLaplaceIPDGInnerCouplingIntegrand
from dune.gdt import LocalIPDGInnerPenaltyIntegrand
from dune.gdt import estimate_combined_inverse_trace_inequality_constant
from dune.gdt import estimate_element_to_intersection_equivalence_constant

from dune.xt.grid import ApplyOnInnerIntersectionsOnce

def assemble_coupling_ops(spaces, ss, nn):
    coupling_grid = dd_grid.coupling_grid(ss, nn) # CouplingGridProvider
    inside_space = spaces[ss]
    outside_space = spaces[nn]
#     sparsity_pattern = make_element_and_intersection_sparsity_pattern(inside_space)
    coupling_op = MatrixOperator(
        coupling_grid,
        inside_space,
        outside_space,
        # ***** which sparsity pattern ******
#          sparsity_pattern
      )
    
    coupling_form = BilinearForm(coupling_grid)
    
    # **** find the correct bilinear form, integrands and filter.  !!! 
    symmetry_factor = 1
    weight = 1
    penalty_parameter= 16
    
    if not penalty_parameter:
        # TODO: check if we need to include diffusion for the coercivity here!
        # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
#         C_G = estimate_element_to_intersection_equivalence_constant(grid)
        # TODO: lapacke missing ! 
#         C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
#         penalty_parameter = C_G *C_M_times_1_plus_C_T
#         if symmetry_factor == 1:
#             penalty_parameter *= 4
    assert penalty_parameter > 0
    
    # grid, local_grid or coupling_grid
    diffusion = GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))
    weight = GridFunction(grid, weight, dim_range=(Dim(d), Dim(d)))
    
    coupling_integrand = LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
    #         LocalIPDGCouplingIntegrand(..., intersection_type=Coupling(coupling_grid))
    penalty_integrand = LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)
    
    local_bilinear_form = LocalCouplingIntersectionIntegralBilinearForm(coupling_integrand + penalty_integrand) 
    
    filter_ = ApplyOnInnerIntersectionsOnce(coupling_grid)
    
    coupling_form += (local_bilinear_form, filter_)
    
    coupling_op.append(coupling_form)
    coupling_op.assemble()
    return coupling_op
```

```python
for ss in range(S):
    for nn in dd_grid.neighbors(ss):
        coupling_ops = assemble_coupling_ops(spaces, ss, nn)
        # additional terms to diagonal
        ops[ss][ss] += coupling_ops[0]
        ops[nn][nn] += coupling_ops[3]
        
        # coupling terms
        if ops[ss][nn] is None:
            ops[ss][nn] = [coupling_ops[1]]
        else:
            ops[ss][nn] += coupling_ops[1]
        if ops[nn][ss] is None:
            ops[nn][ss] = [coupling_ops[2]]
        else:
            ops[nn][ss] += coupling_ops[2]
```

```python
from pymor.operators.block import BlockOperator

block_op = BlockOperator(ops)
```
