# autodfem

A little fem framework I've built for learning mostly.
As I was studying on how automatic differentiation works, I came across the way to do it with Dual Numbers... And so I tried to come up with something to apply it... Why not the Jacobian in Newton's Method. And so I began writing it from what I knew already (and didn't).

It is not production grade, of course: most of it is HIGHLY unoptimized python code, although numpy helps a ton (and do not in some parts due to broadcasting issues). AD is a big help because you don't have to define the tangent stiffness matrix at all... but adds overhead and worsen the time needed to assemble the system.

For now, it only supports Quadrilaterals (although adding simplex wouldn't be too much of an issue I think)

There are some heavy assumptions on this code written here, for applying BCs what is used is a boundary marker function (returns true or false given x,y points), which works for simple meshes mostly... but more complex meshes it sure is (and will) be an issue. One thing I might do it in the future is to actually use markers directly exported from GMSH.

There is a somewhat fragile GMSH support, but some mesh factory functions for structured ones as well.

This project began mostly as I was applying it to Nodal Element (Lagrange) only... So there may be some parts of the code that is very "nodal centered". It somehow became an issue when as I was refactoring some of the code to add other type of elements, in which DOFs are not simple point evaluations. In the element class (and others) there is some naming like "n_nodes"... which actually is supposed to mean n_dofs. I did not change it as most of the code depended on it, so this doublethinking is kinda useful for a lot of cases.

The examples files should explain themselves with their own md files, but the main idea is as follows:

First of all, we must define a mesh and function spaces in our (to be solved) problem. As of now, using either GMSH with linear quadrilaterals (or biquadratic, not serendipity) or the built in mesh generation functions for simple domains will work. Once the mesh is defined, we create a function space, by assigning a mesh to it and the element type as its input.

After, we must define our problem. We will define it through a function with the inputs: function(N, B_x, u_gp, grad_u, x_gp, e)
In order they mean: N the shape functions, B_x the matrix of derivative of the shape functions (for poisson it is the grad(v)), u_gp: it is the value of the field (unknown) AFTER the geometric mapping is applied (so not in reference quadrature space, as _gp might make it seem), grad_u will be for most cases the gradient of u, x_gp is the value of the physical points in the domain (in the case we need to define a rhs, body loads, etc), e will be mostly useless, but was added for very specific purposes

The weak form definition must follow as above. We may now assemble our system of equations. As this project began mostly as to apply AD to something, what was built was with the idea of being residual based for nonlinear problems. That is, the assembler will actually return the residual R(u;v) = 0. This is important, as after defining and assembling the weak form we pass it to the solver (newton raphson solver) with defined boundary conditions. There is, however, a workaround to solve linear problems in linear form instead, as shown in some examples. But anyhow, even when using the Newton solver it should converge in 1 iteration if things are set properly for linear problems.

For post processing, it is done exclusively with Paraview in mind, so we can export vtu files. The function export_vtu will generate a visualization mesh, sampling a few points over each elements, it is not continuous therefore. There are other help functions that can calculate other quantities that depend of the solution itself (as stress) and some helpers to extrapolate/stress recovery the results.