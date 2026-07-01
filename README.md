# autodfem

A little FEM framework I've built for learning mostly.

As I was studying on how automatic differentiation works, I came across the way to do it with [Dual Numbers](https://en.wikipedia.org/wiki/Dual_number)... And so I tried to come up with something to apply it... Why not the Jacobian in Newton's Method. And so I began writing it from what I knew already (and didn't).


## Disclaimer


It is not production grade, of course.

Most of it is **HIGHLY unoptimized Python code**, although NumPy helps a ton (and does not in some parts due to broadcasting issues).

AD is a big help because you don't have to define the tangent stiffness matrix at all... but adds overhead and worsens the time needed to assemble the system.


## Scope

For now, it only supports **quadrilaterals** (although adding simplex should not be too much of an issue).


## Boundary conditions & assumptions

There are some heavy assumptions on this code.

For applying boundary conditions, what is used is a **boundary marker function** (returns true or false given x, y points). This works for simple meshes mostly, but for more complex meshes it will be (and already is) an issue.

One thing I might do in the future is to actually use markers directly exported from Gmsh to identify the boundaries.

There is a somewhat fragile Gmsh support, but also mesh factory functions for structured meshes.

## Naming issues and confusion

This project began mostly when I was applying it to nodal (Lagrange) elements only.

So there may be parts of the code that are very “nodal-centered”.

It became an issue later when refactoring to add other types of elements, where DOFs are not simple point evaluations.

In the element class (and others) there is naming like `n_nodes`... which actually is supposed to mean `n_dofs`.

I did not change it because most of the code depended on it, so this “double-thinking” is still useful in many places.

## Examples

Each example contains its own `README.md` explaining the formulation and how to run it. Github is really problematic on how it deals with latex/math, for example some **bold** letters that should represent vectors in the examples readme's are not rendering properly. So one should proceed with care.

| Example | Description |
|---------|-------------|
| [Poisson](examples/poisson/) | Steady Poisson equation. |
| [Infinite Plate](examples/infinite_plate/) | Linear Elasticity example with an infinite plate approximation. |
| [Elliptic Membrane](examples/eliptic_membrane/) | Linear Elasticity membrane problem. |
| [Lid Driven Cavity](examples/lid_driven/) | Incompressible flow benchmark (with Taylor-Hood Elements). |
| [Mixed Poisson](examples/mixed_poisson/) | Mixed formulation using Raviart-Thomas elements. |
| [Transient Heat](examples/transient_heat/) | Time-dependent heat equation. |
| [Cook's Membrane (Hyperelasticity)](examples/cook_membrane_hyperelasticity/) | Hyperelastic Cook's membrane benchmark. |

## Using it

The examples files should explain themselves with their own markdown files, but the main idea is as follows:

---

### 1. Mesh and function space

First define:
- a mesh
- a function space
- an element type

As of now, you can use:
- Gmsh with linear quadrilaterals (or biquadratic, not serendipity)
- built-in mesh generation functions for simple domains

---

### 2. Problem definition (weak form)

We define the problem through a function:

```python
function(N, B_x, u_gp, grad_u, x_gp, e)
```

Where:

- `N`: shape functions  
- `B_x`: derivative matrix of shape functions (for Poisson it is the grad(v))  
- `u_gp`: value of the field (unknown) AFTER geometric mapping (not reference space)  
- `grad_u`: gradient of u  
- `x_gp`: physical coordinates (used for RHS, body loads, etc.)  
- `e`: element index (mostly unused, added for special cases)

---

### 3. Assembly & solver

The weak form must follow the structure above.

The system is assembled as a **residual formulation**:

$$
R(u; v) = 0
$$

This is important: the assembler returns a residual-based formulation designed for nonlinear problems.

After assembly, the system is passed to a **Newton–Raphson solver** with boundary conditions.

There is also a workaround for linear problems in linear form.

Even when using Newton, linear problems should converge in **1 iteration** if everything is correct.


## Post-processing

Post-processing is done mainly with **ParaView** in mind.

VTU export works by:
- sampling points inside each element
- building a visualization mesh

This results in a discontinuous field representation.

There are also helper functions for:
- derived quantities (stress, etc.)
- extrapolation
- stress recovery
- postprocessing utilities

# To do
If I find time:
- Expose a way to define edge integrals over elements, as it would probably extend easily to DG methods; I do have support for discontinuous elements already as it is used in some mixed formulations in the examples
- Nistche's method for BCs and for it a way to define integrals over the whole (or part) of the boundary... This has to do more with the markers to be exported from gmsh itself instead of simple in,outside evaluations