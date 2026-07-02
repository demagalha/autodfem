import numpy as np
from fem_engine.element import Quad4, Quad9

# Theare are some HEAVY assumptions on all this implementation
# For Neumann BC we are assuming our boundary is straight edged... for the normals (even though quadratic mesh is supported in other parts)
# As of now only Quadrilaterals, linear and quadratic lagrange is supported
# Also: strong enforcement of Dirichlet BC assumes heavily that we are dealing with nodal basis/elements... so kroenecker delta
# Also: I use boundary markers and search with a function the nodes... Ideally it would be better if gmsh or whatever inputs the mesh has the boundaries marked...

class DirichletBC:
    def __init__(self, function_space, value, boundary_marker_func, component=None):
        self.V = function_space
        self.value = value  # Can be a float OR a function: f(x, y)
        self.component = component
        self.boundary_marker = boundary_marker_func
        
        # Pre-compute the unique base DOFs that lie on the boundary
        self.dofs_and_targets = self._find_boundary_dofs()

    def _find_boundary_dofs(self):
        """Routes DOF finding based on the element's mapping type."""
        # Check if the element explicitly identifies as Hdiv
        if getattr(self.V.element, "mapping_type", "H1") == "Hdiv":
            return self._find_boundary_dofs_hdiv()
            
        # --- Existing H1 Logic ---
        # I refactored this last, this is why it is a little messy...
        geom_element = self.V.mesh.geom_element
        ref_coords = self.V.element.get_ref_coords()
        
        constrained = {}
        for e, cell in enumerate(self.V.mesh.cells):
            geom_node_indices = cell[1]
            elem_nodes = self.V.mesh.points[geom_node_indices]
            base_dofs = self.V.cell_dofs[e]
            
            for local_idx, xi in enumerate(ref_coords):
                global_dof = base_dofs[local_idx]
                if global_dof in constrained: continue
                
                N_geom = geom_element.shape_functions(xi)
                pos = N_geom @ elem_nodes
                
                if self.boundary_marker(pos[0], pos[1]):
                    if callable(self.value):
                        val = self.value(pos[0], pos[1])
                    else:
                        val = self.value
                    constrained[global_dof] = val
                    
        if len(constrained) == 0:
            print("WARNING: DirichletBC found ZERO boundary nodes... Check boundary_marker")        
        else:
            print(f"DirichletBC (H1) constrained {len(constrained)} DOFs")
        return constrained


    def _find_boundary_dofs_hdiv(self):
        """Integrates the exact vector flux along boundary edges for H(div) spaces."""
        geom_element = self.V.mesh.geom_element
        
        constrained = {}
        quad_degree = 5 # 3 points is enough for quadratic edges
        
        # Derivatives of the 1D mapping functions with respect to 's'
        ref_tangents = [
            np.array([ 1.0,  0.0]), # Edge 0: Bottom
            np.array([ 0.0,  1.0]), # Edge 1: Right
            np.array([-1.0,  0.0]), # Edge 2: Top
            np.array([ 0.0, -1.0])  # Edge 3: Left
        ]
        
        for e, cell in enumerate(self.V.mesh.cells):
            geom_node_indices = cell[1]
            elem_nodes = self.V.mesh.points[geom_node_indices]
            base_dofs = self.V.cell_dofs[e]
            dof_signs = self.V.get_dof_signs(e)
            
            for bnd_idx in range(self.V.element.n_boundaries):
                # 1. Quick check: Evaluate the midpoint to see if this edge is on the boundary <- this is very fragile.... ideally we'd have the element edges marked in the mesh
                pts_mid, _, _ = geom_element.get_boundary_quadrature(bnd_idx, 1)
                N_geom_mid = geom_element.shape_functions(pts_mid[0])
                pos_mid = N_geom_mid @ elem_nodes
                
                if not self.boundary_marker(pos_mid[0], pos_mid[1]):
                    continue
                
                # 2. Get global DOF and verify it hasn't been constrained yet
                local_dofs = self.V.element.get_facet_dofs(bnd_idx)
                if not local_dofs: continue

                for local_dof in local_dofs:
                    global_dof = base_dofs[local_dof]
                    if global_dof in constrained: continue

                    # 3. Integrate exactly: \int (q . n) * sign * ds
                    sign = dof_signs[local_dof]
                    pts, wgts, _ = geom_element.get_boundary_quadrature(bnd_idx, quad_degree)
                    ref_t = ref_tangents[bnd_idx]

                    dof_integral = 0.0
                    for gp, w in zip(pts, wgts):
                        # Coordinate mapping
                        N_geom = geom_element.shape_functions(gp)
                        pos_gp = N_geom @ elem_nodes

                        # Jacobian mapping
                        B_ref_geom = geom_element.shape_gradients_reference(gp)
                        J = geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)

                        # Exact physical tangent, normal, and differential length
                        phys_t = J @ ref_t
                        detJ_1d = np.linalg.norm(phys_t) # ds
                        normal = np.array([phys_t[1], -phys_t[0]]) / detJ_1d # Outward normal

                        if callable(self.value):
                            #q_exact = np.array(self.value(pos_gp[0], pos_gp[1]))
                            g = self.value(pos_gp[0], pos_gp[1])
                        else:
                            #q_exact = np.array(self.value)
                            g = self.value
                            
                        #dof_integral += np.dot(q_exact, normal) * sign * detJ_1d * w
                        dof_integral += g * sign * detJ_1d * w
                    constrained[global_dof] = dof_integral
                
        if len(constrained) == 0:
            print("WARNING: DirichletBC found ZERO boundary nodes... Check boundary_marker")        
        else:
            print(f"DirichletBC (Hdiv) constrained {len(constrained)} DOFs")
            
        return constrained
    

    def apply(self, R_global, K_global, U_current, method="strong", penalty=1e8, offset=0, is_linear=False):
        if not self.dofs_and_targets:
            return R_global, K_global

        constrained_dofs = []
        target_values = []
        
        for base_dof, target_val in self.dofs_and_targets.items():
            if self.component is None:
                for c in range(self.V.n_components):
                    constrained_dofs.append(base_dof * self.V.n_components + c + offset)
                    target_values.append(target_val)
            else:
                constrained_dofs.append(base_dof * self.V.n_components + self.component + offset)
                target_values.append(target_val)
                
        dofs = np.array(constrained_dofs)
        vals = np.array(target_values)

        if method == "strong":
            if is_linear:
                # --- LINEAR FEM (Elimination Method) ---
                # Preserves matrix symmetry for solvers like CG
                for dof, val in zip(dofs, vals):
                    R_global -= (K_global[:, dof].toarray().ravel() * val)

                K_global[dofs, :] = 0.0
                K_global[:, dofs] = 0.0
                K_global[dofs, dofs] = 1.0
                R_global[dofs] = vals
                
            else:
                # --- NON-LINEAR FEM (Newton-Raphson) ---
                # We are solving J * dU = -R. 
                # Do NOT touch the columns or modify the interior residuals
                K_global[dofs, :] = 0.0
                K_global[dofs, dofs] = 1.0
                # Forces the solver to take a step dU = -(U_current - val)
                # so the next iteration lands exactly on 'val'
                R_global[dofs] = U_current[dofs] - vals
            
        elif method == "penalty":
            for dof in dofs:
                K_global[dof, dof] += penalty
            # K_global[dofs, dofs] += penalty # we could do this if K_global wasnt sparse... we could recovert to dense and then convert to sparse, but rather do as above
            R_global[dofs] += penalty * (U_current[dofs] - vals)
                
        return R_global, K_global

# Use with care... in 2d delta diracs are not an element of H^-1... stress in the region of this load won't converge under refinement
class PointLoadBC:
    def __init__(self, function_space, load_vector, boundary_marker_func):
        self.V = function_space
        self.load_vector = np.array(load_vector)
        self.boundary_marker = boundary_marker_func
        
        # Re-use the coordinate mapping logic to find DOFs
        dummy_dirichlet = DirichletBC(self.V, 0.0, self.boundary_marker)
        self.dofs = list(dummy_dirichlet.dofs_and_targets.keys())

    def apply(self, R_global, K_global, U_current, offset=0):
        if not self.dofs:
            return R_global, K_global
            
        nodes_array = np.array(self.dofs)
        
        for c in range(self.V.n_components):
            dofs = nodes_array * self.V.n_components + c + offset
            R_global[dofs] -= self.load_vector[c]
            
        return R_global, K_global
    

class NeumannBC:
    def __init__(self, function_space, load_vector, boundary_marker_func, quad_degree=2):
        self.V = function_space
        self.load_vector = load_vector
        self.boundary_marker = boundary_marker_func
        self.quad_degree = quad_degree

    def apply(self, R_global, K_global, U_current, offset=0):
        geom_element = self.V.mesh.geom_element

        for e, cell in enumerate(self.V.mesh.cells):
            geom_node_indices = cell[1]
            elem_nodes = self.V.mesh.points[geom_node_indices]
            base_dofs = self.V.cell_dofs[e]
            
            for bnd_idx in range(self.V.element.n_boundaries):
                # 1. Determine if this entire edge is on the boundary using geometry
                _, _, geom_local_nodes = geom_element.get_boundary_quadrature(bnd_idx, 1)
                corner1 = elem_nodes[geom_local_nodes[0]]
                corner2 = elem_nodes[geom_local_nodes[-1]]
                
                if self.boundary_marker(*corner1) and self.boundary_marker(*corner2):
                    
                    # 2. Get the field's integration points and DOF layout for this boundary
                    pts, wgts, local_nodes = self.V.element.get_boundary_quadrature(bnd_idx, self.quad_degree)
                    
                    # Compute physical length of this boundary (straight line approximation)
                    detJ_1d = 0.5 * np.linalg.norm(corner2 - corner1)
                    
                    # Get the global DOFs for the local nodes on this boundary face
                    global_dofs = np.array([base_dofs[n] for n in local_nodes])
                    
                    for gp, weight in zip(pts, wgts):
                        N_field = self.V.element.shape_functions(gp)
                        N_geom = geom_element.shape_functions(gp)
                        pos_gp = N_geom @ elem_nodes
                        
                        if callable(self.load_vector):
                            t_val = np.array(self.load_vector(pos_gp[0], pos_gp[1]))
                        else:
                            t_val = np.array(self.load_vector)
                            
                        # Distribute load to the boundary nodes
                        for c in range(self.V.n_components):
                            dofs = global_dofs * self.V.n_components + c + offset
                            N_bnd = N_field[local_nodes] 
                            R_global[dofs] -= N_bnd * t_val[c] * detJ_1d * weight
                                
        return R_global, K_global