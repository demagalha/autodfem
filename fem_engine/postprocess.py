import numpy as np
import pyvista as pv
import os

from fem_engine.element import Quad4, Quad9, Quad0
from fem_engine.mesh import FunctionSpace

def export_vtu(functionspace, u_vector, filename, field_name="Field", n_vis_pts=None, exact_func=None, show_plot=False):
    """
    Exports a generic FEM field to a VTU file.
    Uses a fully discontinuous visualization mesh. If n_vis_pts is specified, 
    subdivides each macroscopic element into a dense grid of linear sub-quads 
    for high-resolution visualization of high-order/mixed fields.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    mesh = functionspace.mesh
    field_element = functionspace.element
    
    # 1. Instantiate the dummy geometric element based on the mesh cell type
    geom_element = mesh.geom_element
    if mesh.cell_type == "quad":
        default_vtk_type = pv.CellType.QUAD
        default_pts_per_cell = 4
    elif mesh.cell_type == "quad9":
        default_vtk_type = pv.CellType.BIQUADRATIC_QUAD
        default_pts_per_cell = 9
    else:
        raise NotImplementedError(f"Unsupported geometry type: {mesh.cell_type}")

    # 2. Determine reference points and local sub-cell topology
    if n_vis_pts is None:
        # Default behavior: evaluate strictly at the element's geometric nodes
        ref_pts = geom_element.get_ref_coords()
        sub_cells_topo = [list(range(len(ref_pts)))] # Single macroscopic cell
        vtk_type = default_vtk_type
        pts_per_sub_cell = default_pts_per_cell
    else:
        # High-res behavior: generate N x N grid on the reference domain [-1, 1]^2
        s = np.linspace(-1.0, 1.0, n_vis_pts)
        X, Y = np.meshgrid(s, s)
        ref_pts = np.column_stack([X.ravel(), Y.ravel()])
        
        # Build local connectivity for the smaller sub-quads
        sub_cells_topo = []
        for j in range(n_vis_pts - 1):
            for i in range(n_vis_pts - 1):
                n0 = j * n_vis_pts + i
                n1 = n0 + 1
                n2 = n1 + n_vis_pts
                n3 = n0 + n_vis_pts
                sub_cells_topo.append([n0, n1, n2, n3])
                
        vtk_type = pv.CellType.QUAD
        pts_per_sub_cell = 4

    num_macro_cells = len(mesh.cells)
    num_sub_cells = len(sub_cells_topo) * num_macro_cells
    num_total_pts = len(ref_pts) * num_macro_cells
    
    # 3. Check if the mathematical field outputs a vector or a scalar
    is_vector = (functionspace.n_components > 1) or (getattr(field_element, "mapping_type", "") == "Hdiv")

    # 4. Pre-allocate VTK arrays
    points = np.zeros((num_total_pts, 3))
    
    if is_vector:
        # PyVista/VTK expects vector data to be 3D. We then leave Z as 0.
        point_data = np.zeros((num_total_pts, 3)) 
    else:
        point_data = np.zeros(num_total_pts)
        
    vtk_cells = []
    cell_types = np.full(num_sub_cells, vtk_type, dtype=np.uint8)
    
    pt_idx = 0
    for e, cell in enumerate(mesh.cells):
        phys_nodes_idx = cell[1]
        phys_nodes = mesh.points[phys_nodes_idx]
        
        # Get the field DOFs and signs for this specific macro cell
        cell_dofs = functionspace.get_dofs(e)
        cell_signs = functionspace.get_dof_signs(e)
        
        # Reshape the local solution vector properly
        n_nodes = len(functionspace.cell_dofs[e])
        local_u = (u_vector[cell_dofs] * cell_signs).reshape((n_nodes, functionspace.n_components))

        base_pt_idx = pt_idx  # Remember the starting index for this macro cell to append later

        # Evaluate at all chosen reference points
        for xi in ref_pts:
            # GEOMETRIC MAPPING
            N_geom = geom_element.shape_functions(xi)
            x_phys = N_geom @ phys_nodes
            
            # FIELD EVALUATION
            N_ref = field_element.shape_functions(xi)
            
            # Use the geometric element to compute the physical Jacobian
            dN_geom = geom_element.shape_gradients_reference(xi)
            J = geom_element.jacobian(phys_nodes, B_ref=dN_geom)
            detJ = np.linalg.det(J)
            
            # Push reference field values to physical space
            N_phys = field_element.push_forward_values(N_ref, J, detJ)

            # Evaluate the field: u(x) = sum(U_i * N_i(x))
            if getattr(field_element, "mapping_type", "") == "Hdiv":
                u_val = N_phys.T @ local_u
            else:
                u_val = N_phys @ local_u

            # Immediately flatten to a 1D array to kill numpy broadcasting bugs
            u_val = np.ravel(u_val)

            # Calculate error if an exact function is provided
            if exact_func is not None:
                exact_val = np.ravel(exact_func(x_phys[0], x_phys[1]))
                u_val = u_val - exact_val  # Error = u_h - u_ex

            # STORE DATA
            points[pt_idx, :2] = x_phys
            
            if is_vector:
                # Safely insert the 1D array into the X, Y slots (leaving Z=0)
                point_data[pt_idx, :len(u_val)] = u_val
            else:
                point_data[pt_idx] = u_val[0]
                
            pt_idx += 1

        # Append sub-cell connectivity for this macro element
        for sub_topo in sub_cells_topo:
            vtk_cells.append(pts_per_sub_cell)
            vtk_cells.extend([base_pt_idx + n for n in sub_topo])

    # 5. Build the Unstructured Grid and Save
    grid = pv.UnstructuredGrid(vtk_cells, cell_types, points)
    
    # Assign data and set it as the active field for visualization
    grid.point_data[field_name] = point_data
    if is_vector:
        grid.set_active_vectors(field_name)
    else:
        grid.set_active_scalars(field_name)

    grid.save(filename)
    print(f"Exported VTU to {filename}")

    if show_plot:
        grid.plot(scalars=field_name, show_edges=True, cmap="jet")


def project_to_cell_centers(V_source, U_source, derived_func, n_components=1):
    """
    Evaluates a derived function (like stress) at cell centers.
    Returns a Quad0 FunctionSpace and the populated DOF vector.
    """
    # Create target space (piecewise constant)
    V_target = FunctionSpace(V_source.mesh, Quad0(), n_components=n_components)
    U_target = np.zeros(V_target.ndofs)

    # We evaluate at the reference centroid
    centroid = np.array([0.0, 0.0])
    N_ref = V_source.element.shape_functions(centroid)
    B_ref = V_source.element.shape_gradients_reference(centroid)

    # Determine geometry element
    geom_element = V_source.mesh.geom_element

    for e, cell in enumerate(V_source.mesh.cells):
        geom_node_indices = cell[1]
        phys_nodes = V_source.mesh.points[geom_node_indices]

        # Geometric Jacobian at centroid
        B_ref_geom = geom_element.shape_gradients_reference(centroid)
        J = geom_element.jacobian(phys_nodes, B_ref=B_ref_geom)
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # Push forward fields to physical space
        N_phys = V_source.element.push_forward_values(N_ref, J, detJ)
        B_phys = V_source.element.push_forward_derivatives(B_ref, J, invJ, detJ)

        # Fetch local DOFs
        dofs = V_source.get_dofs(e)
        signs = V_source.get_dof_signs(e)
        
        # Reshape to (n_nodes, n_components)
        n_nodes = len(V_source.cell_dofs[e])
        u_loc = (U_source[dofs] * signs).reshape((n_nodes, V_source.n_components))

        # Evaluate u and grad_u at the centroid
        if getattr(V_source.element, "mapping_type", "") == "Hdiv":
            u_gp = N_phys.T @ u_loc
        else:
            u_gp = N_phys @ u_loc
        
        grad_u = B_phys @ u_loc

        # Evaluate the generic user-provided function
        val = derived_func(u_gp, grad_u)

        # Map to the target Quad0 space
        target_dofs = V_target.get_dofs(e)
        U_target[target_dofs] = val

    return V_target, U_target

# before project_to_nodes
def average_at_nodes(V_source, U_source, derived_func, n_components=1):
    """
    Evaluates a derived function at the element corners and averages the results 
    at the global mesh nodes to create a continuous (smoothed) field.
    """
    # Target space is a continuous Quad4 space
    V_target = FunctionSpace(V_source.mesh, Quad4(), n_components=n_components)
    U_target = np.zeros(V_target.ndofs)
    
    # We need to count how many elements share each node to compute the average
    counts = np.zeros(V_target.ndofs)
    
    geom_element = V_source.mesh.geom_element
    
    # We only evaluate at the 4 geometric corners
    ref_nodes = geom_element.get_ref_coords()[:4] 

    for e, cell in enumerate(V_source.mesh.cells):
        geom_node_indices = cell[1]
        phys_nodes = V_source.mesh.points[geom_node_indices]
        
        # Source DOFs
        dofs = V_source.get_dofs(e)
        signs = V_source.get_dof_signs(e)
        n_nodes_source = len(V_source.cell_dofs[e])
        u_loc = (U_source[dofs] * signs).reshape((n_nodes_source, V_source.n_components))
        
        # Target DOFs for this cell (Quad4 continuous global nodes)
        target_dofs = V_target.get_dofs(e)

        # Evaluate at each local corner of the element
        for local_idx, xi in enumerate(ref_nodes):
            # Geometric Mapping
            B_ref_geom = geom_element.shape_gradients_reference(xi)
            J = geom_element.jacobian(phys_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Push forward fields
            N_ref = V_source.element.shape_functions(xi)
            B_ref = V_source.element.shape_gradients_reference(xi)
            
            N_phys = V_source.element.push_forward_values(N_ref, J, detJ)
            B_phys = V_source.element.push_forward_derivatives(B_ref, J, invJ, detJ)

            # Evaluate fields
            if getattr(V_source.element, "mapping_type", "") == "Hdiv":
                u_gp = N_phys.T @ u_loc
            else:
                u_gp = N_phys @ u_loc
            
            grad_u = B_phys @ u_loc

            # Calculate custom math function (e.g., Von Mises)
            val = derived_func(u_gp, grad_u)
            val = np.ravel(val) # Flatten just in case, tired of broadcast dimention bugs
            
            # Find the specific target DOFs for this node
            start = local_idx * n_components
            end = start + n_components
            global_dof_indices = target_dofs[start:end]

            # Accumulate the value and the counter
            U_target[global_dof_indices] += val
            counts[global_dof_indices] += 1.0

    # Average the values at the shared nodes
    counts[counts == 0] = 1.0  # Prevent division by zero just in case
    U_target /= counts

    return V_target, U_target

# before project_gauss_stress_to_nodes_new
def extrapolate_gauss_to_nodes(V_source, U_source, derived_func, n_components=1, quad_degree=None):
    """
    Stress recovery via Least-Squares Gauss Point Extrapolation.
    Target is always Quad4 (corners) for easy visualization.
    Supports any source element (Quad4, Quad9) and arbitrary quadrature degrees.
    """
    # Automatically select a quadrature degree if not provided
    if quad_degree is None:
        quad_degree = 5 if V_source.mesh.cell_type == "quad9" else 2

    # Target space is always Quad4 (corners/vertex only)
    V_target = FunctionSpace(V_source.mesh, Quad4(), n_components=n_components)

    U_target = np.zeros(V_target.ndofs)
    counts = np.zeros(V_target.ndofs)

    geom_element = V_source.mesh.geom_element

    # Build generalized extrapolation matrix
    pts, wgts = geom_element.get_volume_quadrature(quad_degree)
    n_gp = len(pts)

    Nmat = np.zeros((n_gp, 4))
    target_element = Quad4()

    # Evaluate the Quad4 shape functions at the source's Gauss points
    for i, xi in enumerate(pts):
        Nmat[i, :] = target_element.shape_functions(xi)

    # Use the Moore-Penrose pseudo-inverse for Least-Squares fitting.
    # If n_gp == 4, this acts exactly like np.linalg.inv.
    # If n_gp > 4, finds the optimal best-fit Quad4 plane through the points.
    E = np.linalg.pinv(Nmat)

    # Loop over elements
    for e, cell in enumerate(V_source.mesh.cells):

        geom_node_indices = cell[1]
        phys_nodes = V_source.mesh.points[geom_node_indices]

        dofs = V_source.get_dofs(e)
        signs = V_source.get_dof_signs(e)

        n_nodes_source = len(V_source.cell_dofs[e])

        u_loc = (U_source[dofs] * signs).reshape((n_nodes_source, V_source.n_components))

        # Evaluate derived quantity at Gauss points
        gp_values = np.zeros((n_gp, n_components))

        for gp_idx, xi in enumerate(pts):

            B_ref_geom = geom_element.shape_gradients_reference(xi)

            J = geom_element.jacobian(phys_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            N_ref = V_source.element.shape_functions(xi)
            B_ref = V_source.element.shape_gradients_reference(xi)

            N_phys = V_source.element.push_forward_values(N_ref, J, detJ)
            B_phys = V_source.element.push_forward_derivatives(B_ref, J, invJ, detJ)

            if getattr(V_source.element, "mapping_type", "") == "Hdiv":
                u_gp = N_phys.T @ u_loc
            else:
                u_gp = N_phys @ u_loc

            grad_u = B_phys @ u_loc

            val = np.ravel(derived_func(u_gp, grad_u))
            gp_values[gp_idx, :] = val

        # Extrapolate GP values -> nodal values
        nodal_values = E @ gp_values

        # Accumulate into global nodes
        target_dofs = V_target.get_dofs(e)

        # We only accumulate to the 4 corner nodes of the target space
        for local_node in range(4):

            start = local_node * n_components
            end = start + n_components

            global_dof_indices = target_dofs[start:end]

            U_target[global_dof_indices] += nodal_values[local_node]
            counts[global_dof_indices] += 1.0

    counts[counts == 0.0] = 1.0
    U_target /= counts

    return V_target, U_target

# before project_mixed_gauss_stress_to_nodes
def extrapolate_mixed_gauss_to_nodes(V_u, u_sol, V_p, p_sol, derived_func, n_components=1, quad_degree=None):
    """
    Stress recovery for mixed u-p elements via Least-Squares Gauss Point Extrapolation.
    Takes BOTH the displacement space (V_u) and pressure space (V_p) simultaneously.
    """
    if quad_degree is None:
        quad_degree = 5 if V_u.mesh.cell_type == "quad9" else 2

    # Target space is always Quad4 (corners only) for continuous visualization
    V_target = FunctionSpace(V_u.mesh, Quad4(), n_components=n_components)

    U_target = np.zeros(V_target.ndofs)
    counts = np.zeros(V_target.ndofs)

    geom_element = V_u.mesh.geom_element
    pts, wgts = geom_element.get_volume_quadrature(quad_degree)
    n_gp = len(pts)

    # Build generalized extrapolation matrix E
    Nmat = np.zeros((n_gp, 4))
    target_element = Quad4()
    for i, xi in enumerate(pts):
        Nmat[i, :] = target_element.shape_functions(xi)
    E = np.linalg.pinv(Nmat)

    for e, cell in enumerate(V_u.mesh.cells):
        geom_node_indices = cell[1]
        phys_nodes = V_u.mesh.points[geom_node_indices]

        # Fetch Displacement DOFs
        dofs_u = V_u.get_dofs(e)
        signs_u = V_u.get_dof_signs(e)
        n_nodes_u = len(V_u.cell_dofs[e])
        u_loc = (u_sol[dofs_u] * signs_u).reshape((n_nodes_u, V_u.n_components))

        # Fetch Pressure DOFs
        dofs_p = V_p.get_dofs(e)
        signs_p = V_p.get_dof_signs(e)
        n_nodes_p = len(V_p.cell_dofs[e])
        p_loc = (p_sol[dofs_p] * signs_p).reshape((n_nodes_p, V_p.n_components))

        gp_values = np.zeros((n_gp, n_components))

        # Evaluate at each Gauss Point
        for gp_idx, xi in enumerate(pts):
            B_ref_geom = geom_element.shape_gradients_reference(xi)
            J = geom_element.jacobian(phys_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Evaluate Kinematics (grad_u)
            N_ref_u = V_u.element.shape_functions(xi)
            B_ref_u = V_u.element.shape_gradients_reference(xi)
            N_phys_u = V_u.element.push_forward_values(N_ref_u, J, detJ)
            B_phys_u = V_u.element.push_forward_derivatives(B_ref_u, J, invJ, detJ)
            
            u_gp = N_phys_u @ u_loc
            grad_u = B_phys_u @ u_loc

            # Evaluate Pressure (p)
            N_ref_p = V_p.element.shape_functions(xi)
            N_phys_p = V_p.element.push_forward_values(N_ref_p, J, detJ)
            B_ref_p = V_p.element.shape_gradients_reference(xi)
            B_phys_p = V_p.element.push_forward_derivatives(B_ref_p, J, invJ, detJ)
            
            p_gp = N_phys_p @ p_loc
            grad_p = B_phys_p @ p_loc

            # Pass u, grad_u, AND p, grad_p to the user's derived formula
            val = np.ravel(derived_func(u_gp, grad_u, p_gp, grad_p))
            gp_values[gp_idx, :] = val

        # Extrapolate GP values -> nodal values
        nodal_values = E @ gp_values

        # Accumulate into global nodes
        target_dofs = V_target.get_dofs(e)
        for local_node in range(4):
            start = local_node * n_components
            end = start + n_components
            global_dof_indices = target_dofs[start:end]

            U_target[global_dof_indices] += nodal_values[local_node]
            counts[global_dof_indices] += 1.0

    counts[counts == 0.0] = 1.0
    U_target /= counts

    return V_target, U_target

def evaluate_field_at_point(mesh, V, U, x_phys):
    """
    Evaluate FEM field at a physical point using BVH + map_to_reference.
    Works for structured and unstructured meshes (from what I've tested so far).
    """

    candidates = mesh.find_candidates(x_phys)

    for e in candidates:
        cell = mesh.cells[e][1]
        nodes = mesh.points[cell]

        geom_element = mesh.geom_element

        # 1. Map physical point -> reference element
        try:
            xi = geom_element.map_to_reference(nodes, x_phys)
        except RuntimeError:
            continue
        
        # Verify point is actually inside reference element
        tol = 1e-10

        if np.any(xi < -1.0 - tol) or np.any(xi > 1.0 + tol):
            continue

        # 2. Shape functions at reference coord
        N = V.element.shape_functions(xi)

        # 3. Element DOFs
        dofs = V.get_dofs(e)

        u_loc = U[dofs].reshape((V.element.n_nodes, V.n_components))

        # 4. FEM interpolation
        return N @ u_loc

    return np.nan