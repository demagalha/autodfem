import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
import time
import inspect

from fem_engine.dual import jacobian

def _safe_squeeze(x):
    """Squeezes trailing dimension only if it exists and has size 1."""
    if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[-1] == 1:
        return x.squeeze(axis=-1)
    return x

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
        
        # PRECOMPUTE GEOMETRY (PURE FLOATS)
        mapped_data = []
        
        for i in range(len(self.wgts)):
            weight = self.wgts[i]
            gp = self.pts[i]
            N_ref = self.N_cache[i]
            B_ref = self.B_ref_cache[i]
            
            # Geometry Mapping
            B_ref_geom = self.geom_element.shape_gradients_reference(gp)
            J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            # Physical coordinate uses the geometry shape functions
            N_geom = self.geom_element.shape_functions(gp)
            pos_gp = N_geom @ elem_nodes 
            
            # Field Mapping
            N = self.V.element.push_forward_values(N_ref, J, detJ)
            B_x = self.V.element.push_forward_derivatives(B_ref, J, invJ, detJ)
            
            # Store everything required for integration
            mapped_data.append((weight, detJ, pos_gp, N, B_x))
            
        # DEFINE RESIDUAL
        def elem_residual(u_loc_1d):
            # u_loc_1d is natively a DualTensor passed by jacobian
            u_loc = u_loc_1d.reshape((n_nodes_per_elem, self.V.n_components))
            
            # Initialize a DualTensor of zeros to accumulate the residual
            R_loc = u_loc_1d * 0.0 
            
            for (weight, detJ, pos_gp, N, B_x) in mapped_data:
                
                if self.V.element.mapping_type == "Hdiv":
                    u_gp = N.T @ u_loc
                    grad_u = B_x @ u_loc
                else:
                    # H1 mainly
                    u_gp = N @ u_loc
                    grad_u = B_x @ u_loc

                # Due to the way basis fun and stuff are defined, this should fix dimentions mismatch for some
                if self.V.n_components == 1:
                    u_gp = _safe_squeeze(u_gp)
                    grad_u = _safe_squeeze(grad_u)

                if self.expects_state:
                    gp_state = {}
                    for key, grad_key, is_field in param_keys:
                        if is_field:
                            val = local_params[key]
                            val_gp = N @ val
                            grad_val = B_x @ val
                            if self.V.n_components == 1:
                                val_gp = _safe_squeeze(val_gp)
                                grad_val = _safe_squeeze(grad_val)
                            gp_state[key] = val_gp
                            gp_state[grad_key] = grad_val 
                        else:
                            gp_state[key] = local_params[key]
                    
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, e, gp_state)
                else:
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, e)

                R_loc += (integrand * (detJ * weight)).flatten()
                
            return R_loc

        return jacobian(elem_residual, u_loc_guess)

    def assemble(self, U_current, global_params=None):
        if global_params is None:
            global_params = {}

        field_params_def = {}
        scalar_params_def = {}
        param_keys = []

        for key, val in global_params.items():
            if isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], 'get_dofs'):
                V_field, arr = val
                field_params_def[key] = (V_field, arr)
                param_keys.append((key, f'grad_{key}', True))
            elif isinstance(val, np.ndarray):
                field_params_def[key] = (self.V, val)
                param_keys.append((key, f'grad_{key}', True))
            else:
                scalar_params_def[key] = val
                param_keys.append((key, None, False))

        start_time = time.time()
        ndofs = self.V.ndofs
        
        R_global = np.zeros(ndofs)
        
        # Pre-allocate lists for COO assembly
        rows = []
        cols = []
        data = []
        
        for e in range(len(self.V.mesh.cells)):
            dof_indices = self.V.get_dofs(e)
            dof_signs = self.V.get_dof_signs(e)
            n_nodes_per_elem = len(self.V.cell_dofs[e])

            # We pre multiply u by signs... in H1 case dof_signs will be all... 1.0
            u_loc_guess = U_current[dof_indices] * dof_signs

            local_params = {}
            if self.expects_state:
                # Slice using the specific FunctionSpace topo for each field
                for key, (V_field, val_array) in field_params_def.items():
                    f_dofs = V_field.get_dofs(e)
                    f_signs = V_field.get_dof_signs(e)
                    f_n_nodes = len(V_field.cell_dofs[e])
                    local_slice = val_array[f_dofs] * f_signs
                    val = local_slice.reshape((f_n_nodes, V_field.n_components))
                    if V_field.n_components == 1:
                        val = val.squeeze(axis=-1)
                    local_params[key] = val
                    
                for key, val in scalar_params_def.items():
                    local_params[key] = val

            R_loc, K_loc = self.assemble_element(e, u_loc_guess, local_params, param_keys)

            # POST MULTIPLY RESIDUAL AND JACOBIAN BY SIGNS
            R_loc_signed = R_loc * dof_signs
            # Mat mul with diagonal sign matrix
            K_loc_signed = K_loc * np.outer(dof_signs, dof_signs)
            
            # Vectorized COO Data
            # meshgrid generates all index pairs for this element's DOFS instantly
            grid_i, grid_j = np.meshgrid(dof_indices, dof_indices, indexing='ij')
            
            rows.extend(grid_i.flatten())
            cols.extend(grid_j.flatten())
            data.extend(K_loc_signed.flatten())
            
            # Residual is a dense 1D array
            R_global[dof_indices] += R_loc_signed

        # Build COO matrix (automatically sums duplicates), then instantly convert to LIL
        K_global = coo_matrix((data, (rows, cols)), shape=(ndofs, ndofs)).tolil()

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
        if param_keys is None: param_keys = {}

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
        for key, ptype in param_keys.items():
            if ptype == 'single_field':
                V_field, arr = global_params[key]
                dofs = V_field.get_dofs(e)
                signs = V_field.get_dof_signs(e)
                n_nodes = len(V_field.cell_dofs[e])
                local_params[key] = (arr[dofs] * signs).reshape((n_nodes, V_field.n_components))
            elif ptype == 'mixed_field':
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

        # --- GEOMETRY ---
        geom_node_indices = self.V.spaces[0].mesh.cells[e][1]
        elem_nodes = self.V.spaces[0].mesh.points[geom_node_indices]

        # PRECOMPUTE GEOMETRY (PURE FLOATS)
        mapped_data = []
        for k, (gp, w) in enumerate(zip(self.pts, self.wgts)):
            B_ref_geom = self.geom_element.shape_gradients_reference(gp)
            J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            N_geom = self.geom_element.shape_functions(gp)
            pos_gp = N_geom @ elem_nodes

            space_mappings = []
            for i, V in enumerate(self.V.spaces):
                N_ref = self.N_cache[i][k]
                B_ref = self.B_cache[i][k]
                N = V.element.push_forward_values(N_ref, J, detJ)
                B_x = V.element.push_forward_derivatives(B_ref, J, invJ, detJ)
                space_mappings.append((N, B_x))
                
            mapped_data.append((w, detJ, pos_gp, space_mappings))

        def elem_residual(U_loc_flat):
            # U_loc_flat is a natively passed DualTensor
            cursor = 0
            locals_split = []

            for i, V in enumerate(self.V.spaces):
                n_nodes = len(V.cell_dofs[e])
                size = n_nodes * V.n_components

                chunk = U_loc_flat[cursor:cursor+size]
                locals_split.append(chunk.reshape((n_nodes, V.n_components)))
                cursor += size

            # Initialize a DualTensor of zeros with the exact same shape/variables as the input
            R_loc = U_loc_flat * 0.0

            for (w, detJ, pos_gp, space_mappings) in mapped_data:
                mapped = []
                for i, V in enumerate(self.V.spaces):
                    N, B_x = space_mappings[i]
                    u_loc = locals_split[i]
                    
                    if V.element.mapping_type == "Hdiv":
                        u_gp = N.T @ u_loc
                        grad_u = B_x @ u_loc
                    else:
                        u_gp = N @ u_loc
                        grad_u = B_x @ u_loc

                    if V.n_components == 1:
                        u_gp = _safe_squeeze(u_gp)
                        grad_u = _safe_squeeze(grad_u)

                    mapped.append((N, B_x, u_gp, grad_u))
                    
                # --- STATE HANDLING --- # This is somewhat very fragile... beware
                if self.expects_state:
                    gp_state = {}
                    for key, ptype in param_keys.items():
                        if ptype == 'single_field':
                            N, B_x = space_mappings[0]
                            u_loc_state = local_params[key]
                            val = N @ u_loc_state
                            grad_val = B_x @ u_loc_state
                            if val.shape[-1] == 1:
                                val = val.squeeze(axis=-1)
                                grad_val = grad_val.squeeze(axis=-1)
                            gp_state[key] = val
                            gp_state[f'grad_{key}'] = grad_val
                        elif ptype == 'mixed_field':
                            mapped_state = []
                            for i, V in enumerate(self.V.spaces):
                                N, B_x = space_mappings[i]
                                u_loc_state = local_params[key][i]
                                if V.element.mapping_type == "Hdiv":
                                    val = N.T @ u_loc_state
                                    grad_val = B_x @ u_loc_state
                                else:
                                    val = N @ u_loc_state
                                    grad_val = B_x @ u_loc_state
                                if V.n_components == 1:
                                    val = val.squeeze(axis=-1)
                                    grad_val = grad_val.squeeze(axis=-1)
                                mapped_state.append((val, grad_val))
                            gp_state[key] = mapped_state
                        else:
                            gp_state[key] = local_params[key]
                            
                    contribs = self.weak_form(mapped, pos_gp, e, gp_state)
                else:
                    contribs = self.weak_form(mapped, pos_gp, e)

                cursor = 0
                for i, contrib in enumerate(contribs):
                    # We must flatten the DualTensor to 1D and multiply by integration weights
                    flat_contrib = contrib.flatten() * (detJ * w)
                    size = flat_contrib.real.size
                    R_loc[cursor:cursor+size] += flat_contrib
                    cursor += size

            return R_loc.flatten()

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
        param_keys = {}
        for key, val in global_params.items():
            if isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], 'get_dofs'):
                param_keys[key] = 'single_field'
            elif isinstance(val, np.ndarray) and val.size == self.V.ndofs:
                param_keys[key] = 'mixed_field'
            else:
                param_keys[key] = 'scalar'

        ndofs = self.V.ndofs
        R = np.zeros(ndofs)
        
        # Pre-allocate lists for COO assembly
        rows = []
        cols = []
        data = []

        for e in range(len(self.V.spaces[0].mesh.cells)):
            R_loc, K_loc, dofs = self.assemble_element(e, U, global_params, param_keys)

            # Vectorized COO Data
            grid_i, grid_j = np.meshgrid(dofs, dofs, indexing='ij')
            
            rows.extend(grid_i.flatten())
            cols.extend(grid_j.flatten())
            # In MixedAssembler, K_loc is already signed in assemble_element
            data.extend(K_loc.flatten())
            
            R[dofs] += R_loc

        # Build COO matrix (automatically sums duplicates), then instantly convert to LIL
        K = coo_matrix((data, (rows, cols)), shape=(ndofs, ndofs)).tolil()

        return R, K
    
def assemble_scalar(function_space, integrand, u_sol=None, quad_degree=2, global_params=None):
    """
    Universally integrates a scalar value over the mesh.
    If u_sol is provided, it pushes forward the FE field and passes it to the integrand.
    global_params will pass secondary fields (state variables) to the integrands
    """

    if global_params is None:
        global_params = {}

    sig = inspect.signature(integrand)
    expects_state = 'state' in sig.parameters

    field_params_def = {}
    scalar_params_def = {}
    if expects_state:
        for key, val in global_params.items():
            if isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], 'get_dofs'):
                field_params_def[key] = val
            elif isinstance(val, np.ndarray):
                field_params_def[key] = (function_space, val)
            else:
                scalar_params_def[key] = val

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

        local_params = {}
        if expects_state:
            for key, (V_field, val_array) in field_params_def.items():
                f_dofs = V_field.get_dofs(e)
                f_signs = V_field.get_dof_signs(e)
                f_n_nodes = len(V_field.cell_dofs[e])
                val = (val_array[f_dofs] * f_signs).reshape((f_n_nodes, V_field.n_components))
                local_params[key] = _safe_squeeze(val) if V_field.n_components == 1 else val

            for key, val in scalar_params_def.items():
                local_params[key] = val
            
        cell_value = 0.0
        
        for k, (gp, w) in enumerate(zip(pts, wgts)):
            # Geometry mapping
            B_ref_geom = geom_element.shape_gradients_reference(gp)
            J = geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            
            N_geom = geom_element.shape_functions(gp)
            pos_gp = N_geom @ elem_nodes
            
            # Field Mapping
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

                if function_space.n_components == 1:
                    u_gp = _safe_squeeze(u_gp)

                grad_u = dN @ u_loc

                if function_space.n_components == 1:
                    grad_u = _safe_squeeze(grad_u)

            if expects_state:
                gp_state = {}
                for key, val in local_params.items():
                    if key in field_params_def:
                        V_field = field_params_def[key][0]
                        N_state_ref = V_field.element.shape_functions(gp)
                        N_state = V_field.element.push_forward_values(N_state_ref, J, detJ)
                        
                        if getattr(V_field.element, "mapping_type", "") == "Hdiv":
                            val_gp = N_state.T @ val
                        else:
                            val_gp = N_state @ val
                            
                        if V_field.n_components == 1:
                            val_gp = _safe_squeeze(val_gp)
                        gp_state[key] = val_gp
                    else:
                        gp_state[key] = val

                f_val = integrand(u_gp, grad_u, pos_gp, e, gp_state)
            else:
                f_val = integrand(u_gp, grad_u, pos_gp, e)
            
            cell_value += f_val * detJ * w
             
        value += cell_value
        
    return value