import numpy as np
from scipy.sparse import lil_matrix
import time
import inspect

from fem_engine.dual import jacobian

class Assembler:
    def __init__(self, function_space, weak_form_integrand, quad_degree=2):
        self.V = function_space
        self.weak_form = weak_form_integrand
        self.quad_degree = quad_degree
        
        sig = inspect.signature(self.weak_form)
        self.expects_state = 'state' in sig.parameters

        # geometry element from mesh
        self.geom_element = self.V.mesh.geom_element

        self._setup_quadrature_cache()

    def _setup_quadrature_cache(self):
        element = self.V.element
        self.pts, self.wgts = element.get_volume_quadrature(self.quad_degree)
        
        self.N_cache = []
        self.B_ref_cache = []
        
        for gp in self.pts:
            self.N_cache.append(element.shape_functions(gp))
            self.B_ref_cache.append(element.shape_gradients_reference(gp))

    def assemble_element(self, e, u_loc_guess, local_params=None, param_keys=None):
        n_nodes_per_elem = len(self.V.cell_dofs[e])
        
        # Nodes that define the geometry layout
        geom_node_indices = self.V.mesh.cells[e][1]
        elem_nodes = self.V.mesh.points[geom_node_indices]
        
        def elem_residual(u_loc_1d):
            R_loc_tensor = np.zeros((n_nodes_per_elem, self.V.n_components), dtype=object)
            u_loc = np.array(u_loc_1d, dtype=object).reshape((n_nodes_per_elem, self.V.n_components))

            for i in range(len(self.wgts)):
                weight = self.wgts[i]
                gp = self.pts[i]
                N_ref = self.N_cache[i]
                B_ref = self.B_ref_cache[i]
                
                # --- GEOMETRY ---
                B_ref_geom = self.geom_element.shape_gradients_reference(gp)
                J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                
                # Physical coordinate uses the geometry shape functions
                N_geom = self.geom_element.shape_functions(gp)
                pos_gp = N_geom @ elem_nodes 
                
                # --- FIELDS ---

                N = self.V.element.push_forward_values(N_ref, J, detJ)
                B_x = self.V.element.push_forward_derivatives(B_ref, J, invJ, detJ)

                # now a little change to handle RT as well...
                if self.V.element.mapping_type == "Hdiv":
                    # N is (4, 2). We want a (2,) vector out.
                    # We transpose N to (2, 4) and multiply by u_loc (4, 1) -> (2, 1)
                    u_gp = N.T @ u_loc

                    # B_x for Hdiv is the divergence (4,)
                    # grad_u becomes the scalar divergence at integration point
                    grad_u = B_x @ u_loc
                else:
                    # H1 mainly
                    u_gp =  N @ u_loc
                    grad_u = B_x @ u_loc

                if self.expects_state:
                    gp_state = {}
                    for key, grad_key, is_field in param_keys:
                        if is_field:
                            val = local_params[key]
                            gp_state[key] = N @ val
                            gp_state[grad_key] = B_x @ val 
                        else:
                            gp_state[key] = local_params[key]
                    
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, e, gp_state)
                else:
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, e)

                R_loc_tensor += integrand * detJ * weight
                
            return R_loc_tensor.flatten()

        return jacobian(elem_residual, u_loc_guess)

    def assemble(self, U_current, global_params=None):
        if global_params is None: 
            global_params = {}

        param_keys = []
        for key, val in global_params.items():
            if isinstance(val, np.ndarray):
                param_keys.append((key, f'grad_{key}', True)) 
            else:
                param_keys.append((key, None, False))

        start_time = time.time()
        ndofs = self.V.ndofs
        K_global = lil_matrix((ndofs, ndofs))
        R_global = np.zeros(ndofs)
        
        for e in range(len(self.V.mesh.cells)):
            dof_indices = self.V.get_dofs(e)
            dof_signs = self.V.get_dof_signs(e)
            n_nodes_per_elem = len(self.V.cell_dofs[e])

            # We pre multiply u by signs... in H1 case dof_signs will be all... 1.0
            u_loc_guess = U_current[dof_indices] * dof_signs

            local_params = {}
            if self.expects_state:
                for key, _, is_field in param_keys:
                    if is_field:
                        # We also pre multiply historical fields by signs
                        local_slice = global_params[key][dof_indices] * dof_signs
                        local_params[key] = local_slice.reshape((n_nodes_per_elem, self.V.n_components))
                    else:
                        local_params[key] = global_params[key]

            R_loc, K_loc = self.assemble_element(e, u_loc_guess, local_params, param_keys)

            # POST MULTIPLY RESIDUAL AND JACOBIAN BY SIGNS
            R_loc_signed = R_loc * dof_signs
            # Mat mul with diagonal sign matrix
            K_loc_signed = K_loc * np.outer(dof_signs, dof_signs)
            
            for i in range(len(dof_indices)):
                R_global[dof_indices[i]] += R_loc_signed[i]
                for j in range(len(dof_indices)):
                    K_global[dof_indices[i], dof_indices[j]] += K_loc_signed[i, j]

        print(f"Assembly took {time.time() - start_time:.3f} seconds")
        return R_global, K_global


class MixedAssembler:
    def __init__(self, mixed_space, weak_form, quad_degree=2):
        self.V = mixed_space
        self.weak_form = weak_form
        self.quad_degree = quad_degree

        sig = inspect.signature(self.weak_form)
        self.expects_state = 'state' in sig.parameters

        # Identify geometry element from master mesh
        self.geom_element = self.V.spaces[0].mesh.geom_element

        element = self.V.spaces[0].element
        self.pts, self.wgts = element.get_volume_quadrature(quad_degree)

        self.N_cache = []
        self.B_cache = []

        for V in self.V.spaces:
            N_list, B_list = [], []
            for gp in self.pts:
                N_list.append(V.element.shape_functions(gp))
                B_list.append(V.element.shape_gradients_reference(gp))
            self.N_cache.append(N_list)
            self.B_cache.append(B_list)
    
    def assemble_element(self, e, U_global, global_params=None, param_keys=None):
        if global_params is None: global_params = {}
        if param_keys is None: param_keys = []

        field_dofs = []
        field_values = []
        field_signs = []

        # Gather local fields using topological method
        for i, V in enumerate(self.V.spaces):
            dofs = V.get_dofs(e) + self.V.offsets[i]
            signs = V.get_dof_signs(e)
            field_dofs.append(dofs)
            field_signs.append(signs)

            n_nodes = len(V.cell_dofs[e])
            val = U_global[dofs].reshape((n_nodes, V.n_components))
            field_values.append(val)

        all_dofs = np.concatenate(field_dofs)
        all_signs = np.concatenate(field_signs)

        # Slice the historical parameter fields (like u_old)
        local_params = {}
        for key, is_field in param_keys:
            if is_field:
                val_global = global_params[key]
                chunks = []
                for i, V in enumerate(self.V.spaces):
                    dofs = V.get_dofs(e) + self.V.offsets[i]
                    n_nodes = len(V.cell_dofs[e])
                    chunk = val_global[dofs].reshape((n_nodes, V.n_components))
                    chunks.append(chunk)
                local_params[key] = chunks
            else:
                local_params[key] = global_params[key]

        def elem_residual(U_loc_flat):
            U_loc_flat = np.array(U_loc_flat, dtype=object)
            cursor = 0
            locals_split = []

            for i, V in enumerate(self.V.spaces):
                n_nodes = len(V.cell_dofs[e])
                size = n_nodes * V.n_components

                chunk = U_loc_flat[cursor:cursor+size]
                locals_split.append(chunk.reshape((n_nodes, V.n_components)))
                cursor += size

            R_loc = np.zeros_like(U_loc_flat, dtype=object)

            for k, (gp, w) in enumerate(zip(self.pts, self.wgts)):
                
                # --- GEOMETRY ---
                geom_node_indices = self.V.spaces[0].mesh.cells[e][1]
                elem_nodes = self.V.spaces[0].mesh.points[geom_node_indices]

                B_ref_geom = self.geom_element.shape_gradients_reference(gp)
                J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                N_geom = self.geom_element.shape_functions(gp)
                pos_gp = N_geom @ elem_nodes

                # --- FIELDS ---
                mapped = []
                for i, V in enumerate(self.V.spaces):
                    N_ref = self.N_cache[i][k]
                    B_ref = self.B_cache[i][k]

                    # Push forward using the element's specific mapping rules
                    N = V.element.push_forward_values(N_ref, J, detJ)
                    B_x = V.element.push_forward_derivatives(B_ref, J, invJ, detJ)
                    
                    u_loc = locals_split[i]
                    
                    if V.element.mapping_type == "Hdiv":
                        u_gp = N.T @ u_loc
                        grad_u = B_x @ u_loc
                    else:
                        u_gp = N @ u_loc
                        grad_u = B_x @ u_loc

                    mapped.append((N, B_x, u_gp, grad_u))
                # --- STATE HANDLING --- # This is somewhat very fragile... beware
                if self.expects_state:
                    gp_state = {}
                    for key, is_field in param_keys:
                        if is_field:
                            mapped_state = []
                            for i, V in enumerate(self.V.spaces):
                                N_ref = self.N_cache[i][k]
                                B_ref = self.B_cache[i][k]

                                # Push forward using the element's specific mapping rules
                                N = V.element.push_forward_values(N_ref, J, detJ)
                                B_x = V.element.push_forward_derivatives(B_ref, J, invJ, detJ)

                                u_loc_state = local_params[key][i]

                                if V.element.mapping_type == "Hdiv":
                                    u_gp_state = N.T @ u_loc_state
                                    grad_u_state = B_x @ u_loc_state
                                else:
                                    u_gp_state = N @ u_loc_state
                                    grad_u_state = B_x @ u_loc_state
                                
                                # Appends a tuple matching the current field map
                                mapped_state.append((u_gp_state, grad_u_state))
                            gp_state[key] = mapped_state
                        else:
                            gp_state[key] = local_params[key]
                            
                    contribs = self.weak_form(mapped, pos_gp, e, gp_state)
                else:
                    contribs = self.weak_form(mapped, pos_gp, e)

                cursor = 0
                for i, contrib in enumerate(contribs):
                    size = contrib.size
                    R_loc[cursor:cursor+size] += contrib.flatten() * detJ * w
                    cursor += size

            return R_loc

        # Pre multiply by sign
        U_loc = U_global[all_dofs] * all_signs
        # Now we pass the pre multiplied U_loc to the AD engine
        R_loc, K_loc = jacobian(elem_residual, U_loc)

        # POST MULTIPLY
        R_loc_signed = R_loc * all_signs
        K_loc_signed = K_loc * np.outer(all_signs, all_signs)

        return R_loc_signed, K_loc_signed, all_dofs
    
    def assemble(self, U, global_params=None):
        if global_params is None:
            global_params = {}

        # Detect which params are global fields (like U_old) vs scalars (like dt)
        param_keys = []
        for key, val in global_params.items():
            if isinstance(val, np.ndarray) and val.size == self.V.ndofs:
                param_keys.append((key, True))
            else:
                param_keys.append((key, False))

        ndofs = self.V.ndofs
        R = np.zeros(ndofs)
        K = lil_matrix((ndofs, ndofs))

        for e in range(len(self.V.spaces[0].mesh.cells)):
            R_loc, K_loc, dofs = self.assemble_element(e, U, global_params, param_keys)

            for i in range(len(dofs)):
                R[dofs[i]] += R_loc[i]
                for j in range(len(dofs)):
                    K[dofs[i], dofs[j]] += K_loc[i, j]

        return R, K
    
def assemble_scalar(function_space, integrand, u_sol=None, quad_degree=2):
    """
    Universally integrates a scalar value over the mesh.
    If u_sol is provided, it pushes forward the FE field and passes it to the integrand.
    """
    geom_element = function_space.mesh.geom_element
        
    pts, wgts = geom_element.get_volume_quadrature(quad_degree)
    
    # Cache reference shape functions
    N_cache = [function_space.element.shape_functions(gp) for gp in pts]
    dN_cache = [function_space.element.shape_gradients_reference(gp) for gp in pts]
    
    value = 0.0
    
    # Loop over all cells
    for e, cell in enumerate(function_space.mesh.cells):
        geom_node_indices = cell[1]
        elem_nodes = function_space.mesh.points[geom_node_indices]
        
        # --- Pre-process DOFs if a solution vector is provided ---
        u_loc = None
        if u_sol is not None:
            dofs = function_space.get_dofs(e)
            signs = function_space.get_dof_signs(e) # CRITICAL for RT fluxes!
            n_nodes = len(function_space.cell_dofs[e]) ## i aded this last
            u_loc = (u_sol[dofs] * signs).reshape((n_nodes, function_space.n_components))
            
        cell_value = 0.0
        
        for k, (gp, w) in enumerate(zip(pts, wgts)):
            # Geometry mapping
            B_ref_geom = geom_element.shape_gradients_reference(gp)
            J = geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            N_geom = geom_element.shape_functions(gp)
            pos_gp = N_geom @ elem_nodes
            
            # --- Field Mapping ---
            u_gp = None
            grad_u = None
            if u_loc is not None:
                N_ref = N_cache[k]
                dN_ref = dN_cache[k]

                N = function_space.element.push_forward_values(N_ref, J, detJ)
                dN = function_space.element.push_forward_derivatives(dN_ref, J, invJ, detJ)
                
                # Check mapping type
                if getattr(function_space.element, "mapping_type", "") == "Hdiv":
                    u_gp = N.T @ u_loc  # Vector flux evaluation
                else:
                    u_gp = N @ u_loc    # Scalar evaluation

                #fixing dimensions, if I have time I will clean a lot of numpy dimension mismatch... but now now
                if u_gp.size == 1:
                    u_gp = u_gp.item()
                else:
                    u_gp = u_gp.flatten()

                grad_u = dN @ u_loc
                    
            # --- Evaluate Integrand ---
            f_val = integrand(u_gp, grad_u, pos_gp, e)
            
            cell_value += f_val * detJ * w
             
        value += cell_value
        
    return value