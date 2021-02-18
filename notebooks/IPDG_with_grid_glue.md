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
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
macro_grid = make_cube_grid(Dim(d), Cube(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])

print(f'grid has {macro_grid.size(0)} elements, {macro_grid.size(d - 1)} edges and {macro_grid.size(d)} vertices')
```

Now we can use this grid as a macro grid for a dd grid.

```python
dd_grid = make_cube_dd_grid(macro_grid, 2)
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
                      LocalElementIntegralBilinearForm)


def assemble_local_op(grid, space, d):
    a_h = MatrixOperator(grid, source_space=space, range_space=space,
                         sparsity_pattern=make_element_sparsity_pattern(space))
    a_form = BilinearForm(grid)
    a_form += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GridFunction(grid, kappa, dim_range=(Dim(d), Dim(d)))))
    a_h.append(a_form)
    a_h.assemble()
    return a_h
```

```python
ops = np.empty((S, S), dtype=object)
```

```python
for ss in range(S):
    space = spaces[ss]
    grid = dd_grid.local_grid(ss)
    grid = grids[ss]
    ops[ss,ss] = assemble_local_op(grid, space, d)
```

```python
def assemble_coupling_ops(spaces, ss, nn):
    coupling_grid = dd_grid.coupling_grid(ss, nn) # CouplingGridProvider
    inside_space = spaces[ss]
    outside_space = spaces[nn]
    coupling_op = MatrixOperator(coupling_grid,
        inside_space,
        outside_space,
        # ***** which sparsity pattern ******
        sparsity_pattern=make_element_and_intersection_sparsity_pattern(inside_space)
    )
    coupling_form = BilinearForm(coupling_grid)
    # **** find the correct bilinear form, integrands and filter.  !!! 
    coupling_form += (LocalCouplingIntersectionIntegralBilinearForm(
                    LocalLaplaceIPDGInnerCouplingIntegrand(symmetry_factor, diffusion, weight)
                    + LocalIPDGInnerPenaltyIntegrand(penalty_parameter, weight)),
                ApplyOnInnerIntersectionsOnce(grid))
    coupling_op.append(coupling_form)
#         LocalIPDGCouplingIntegrand(..., intersection_type=Coupling(coupling_grid))
#     )
    coupling_op.assemble()
    return coupling_op
```

```python
for ss in range(S):
    for nn in dd_grid.neighbors(ss):
        print(nn)
        ops[ss][nn] = assemble_coupling_ops(spaces, ss, nn)
```

```python
from pymor.operators.block import BlockOperator

block_op = BlockOperator(ops)
```
