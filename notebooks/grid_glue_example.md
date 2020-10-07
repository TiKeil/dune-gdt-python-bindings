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

## 1: creating functions

We'll work in 2d in this tutorial since scalar functions in 2d can be best visualized within the notebook.
Let's set up a 2d grid first, as seen in other tutorials and examples.

```python
from dune.xt.grid import Dim, Cube, Simplex, make_cube_grid, make_cube_dd_grid
from dune.xt.functions import ConstantFunction, ExpressionFunction, GridFunction as GF

d = 2
omega = ([0, 0], [1, 1])
grid = make_cube_grid(Dim(d), Cube(), lower_left=omega[0], upper_right=omega[1], num_elements=[2, 2])
dd_grid = make_cube_dd_grid(grid, 2)

print(f'grid has {grid.size(0)} elements, {grid.size(d - 1)} edges and {grid.size(d)} vertices')
```

```python
print(dd_grid)
```

```python
print(dd_grid.dimension)
dir(dd_grid)
```

```python
print(dd_grid.local_grid(0))
print(dd_grid.local_grid(0).centers())
local_grid = dd_grid.local_grid(0)
```

```python
print(grid)
print(grid.max_level)
print(grid.size(0))
dir(grid)
```

```python
print(grid.centers())
```

## 1.1: using the `ConstantFunction`

For constant functions.

```python
from dune.xt.functions import ConstantFunction

alpha = 0.5
f = ConstantFunction(dim_domain=Dim(d), dim_range=Dim(1), value=[alpha], name='f')
# note that we have to provide [alpha], since scalar constant functions expect a vector of length 1

A = [[1, 0], [0, 1]]
g = ConstantFunction(dim_domain=Dim(d), dim_range=(Dim(d), Dim(d)), value=A, name='g')
```

## 1.2: using the `ExpressionFunction`

For functions given by an expression, where we have to specify the polynomial order of the expression (or the approximation order for non-polynomial functions).

Note that if the name of the variable is `Foo`, the components `Foo[0]`, ... `Foo[d - 1]` are availabble to be used in the expression.

```python
h = ExpressionFunction(dim_domain=Dim(d), variable='x', order=10, expression='(0.5 - x[0])^2 * (0.5 - x[1])^2',
                       gradient_expressions=['-2 * (0.5 - x[0]) * (0.5 - x[1])^2', '-2 * (0.5 - x[0])^2 * (0.5 - x[1])'], name='h')
```

## 1.3: discrete functions

Which often result from a discretization scheme, see the *tutorial on continuous Finite Elements for the stationary heat equation*.

```python
from dune.gdt import DiscontinuousLagrangeSpace, DiscreteFunction

V_h = DiscontinuousLagrangeSpace(grid, order=1)
v_h = DiscreteFunction(V_h, name='v_h')

print(v_h.dofs.vector.sup_norm())
```

# 2: visualizing functions

## 2.1: visualizing scalar functions in 2d

We can easily visualize functions mapping from $\mathbb{R}^2 \to \mathbb{R}$. Internally, this is achieved by writing a vtk file to disk and displaying the file using K3D.

```python
from dune.xt.functions import visualize_function

_ = visualize_function(h, grid)
```

**Note**: since functions are always visualized as piecewise linears on each grid element, `dune-grid` supports to write functions on a virtually refined grid, which may be an improvement for higher order data functions. To see this, let us have a look at the rather coarse grid consisting of two simplices:

```python
_ = visualize_function(h, grid, subsampling=True)
```

```python
_ = visualize_function(h, local_grid, subsampling=True)
```

## visualizing the grid

```python
from dune.xt.grid import visualize_grid

_ = visualize_grid(local_grid)
```

```python
_ = visualize_grid(grid)
```
