import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
import time
import inspect

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

class Assembler:
    def __init__(self, function_space, weak_form_integrand, quad_degree=2):
        self.V = function_space
        self.weak_form = weak_form_integrand
        self.quad_degree = quad_degree
        
        sig = inspect.signature(self.weak_form)
        self.expects_state = 'state' in sig.parameters
        self.geom_element = self.V.mesh.geom_element

        self._setup_quadrature_cache()
        self._precompute_geometry()
        self._setup_jax_kernel()
        self._jit_vmap_kernel = None

    def _setup_quadrature_cache(self):
        element = self.V.element
        self.pts, self.wgts = element.get_volume_quadrature(self.quad_degree)
        self.N_cache = [element.shape_functions(gp) for gp in self.pts]
        self.B_ref_cache = [element.shape_gradients_reference(gp) for gp in self.pts]

    def _precompute_geometry(self):
        n_elems = len(self.V.mesh.cells)
        n_quad = len(self.wgts)
        
        detJ_all, pos_gp_all, N_all, B_x_all = [], [], [], []
        self.dof_indices_all, self.dof_signs_all = [], []

        for e in range(n_elems):
            geom_node_indices = self.V.mesh.cells[e][1]
            elem_nodes = self.V.mesh.points[geom_node_indices]
            
            self.dof_indices_all.append(self.V.get_dofs(e))
            self.dof_signs_all.append(self.V.get_dof_signs(e))

            detJ_e, pos_gp_e, N_e, B_x_e = [], [], [], []
            for i in range(n_quad):
                gp = self.pts[i]
                B_ref_geom = self.geom_element.shape_gradients_reference(gp)
                J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                
                N_geom = self.geom_element.shape_functions(gp)
                pos_gp_e.append(N_geom @ elem_nodes)
                detJ_e.append(detJ)
                N_e.append(self.V.element.push_forward_values(self.N_cache[i], J, detJ))
                B_x_e.append(self.V.element.push_forward_derivatives(self.B_ref_cache[i], J, invJ, detJ))

            detJ_all.append(detJ_e)
            pos_gp_all.append(pos_gp_e)
            N_all.append(N_e)
            B_x_all.append(B_x_e)

        self.detJ_jax = jnp.array(detJ_all)
        self.pos_gp_jax = jnp.array(pos_gp_all)
        self.N_jax = jnp.array(N_all)
        self.B_x_jax = jnp.array(B_x_all)
        self.w_jax = jnp.array(self.wgts)

        self.dof_indices_all = np.array(self.dof_indices_all)
        self.dof_signs_all = np.array(self.dof_signs_all)

    def _setup_jax_kernel(self):
        mapping_type = self.V.element.mapping_type
        n_components = self.V.n_components
        n_quad = len(self.wgts)

        def _core_residual_single(u_loc_1d, w_arr, detJ_arr, pos_gp_arr, N_arr, B_x_arr, field_params, scalar_params):
            u_loc = u_loc_1d.reshape((-1, n_components))
            R_loc = jnp.zeros_like(u_loc_1d)

            for i in range(n_quad):
                w, detJ, pos_gp, N, B_x = w_arr[i], detJ_arr[i], pos_gp_arr[i], N_arr[i], B_x_arr[i]

                if mapping_type == "Hdiv":
                    u_gp = jnp.dot(N.T, u_loc)
                    grad_u = jnp.dot(B_x, u_loc)
                else:
                    u_gp = jnp.dot(N, u_loc)
                    grad_u = jnp.dot(B_x, u_loc)

                if self.expects_state:
                    gp_state = {}
                    for key, val in field_params.items():
                        gp_state[key] = jnp.dot(N, val)
                        gp_state[f'grad_{key}'] = jnp.dot(B_x, val)
                    for key, val in scalar_params.items():
                        gp_state[key] = val
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, 0, gp_state)
                else:
                    integrand = self.weak_form(N, B_x, u_gp, grad_u, pos_gp, 0)

                R_loc = R_loc + (integrand * detJ * w).flatten()
            return R_loc

        _compute_jac_single = jax.jacfwd(_core_residual_single, argnums=0)

        def val_and_jac_single(u, w, dJ, pos, N, B, f_p, s_p):
            return _core_residual_single(u, w, dJ, pos, N, B, f_p, s_p), _compute_jac_single(u, w, dJ, pos, N, B, f_p, s_p)

        self._val_and_jac_single = val_and_jac_single

    def assemble(self, U_current, global_params=None):
        start_time = time.time()
        if global_params is None: global_params = {}

        u_loc_all = U_current[self.dof_indices_all] * self.dof_signs_all

        field_params, scalar_params = {}, {}
        n_nodes_per_elem = len(self.V.cell_dofs[0])

        if self.expects_state:
            n_elems = len(self.V.mesh.cells)
            for key, val in global_params.items():
                if isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], 'get_dofs'):
                    V_field, arr = val
                    f_dofs = np.array([V_field.get_dofs(e) for e in range(n_elems)])
                    f_signs = np.array([V_field.get_dof_signs(e) for e in range(n_elems)])
                    f_n_nodes = len(V_field.cell_dofs[0])
                    sliced = arr[f_dofs] * f_signs
                    field_params[key] = sliced.reshape((-1, f_n_nodes, V_field.n_components))
                elif isinstance(val, np.ndarray):
                    sliced = val[self.dof_indices_all] * self.dof_signs_all
                    n_nodes_per_elem = len(self.V.cell_dofs[0])
                    field_params[key] = sliced.reshape((-1, n_nodes_per_elem, self.V.n_components))
                else:
                    scalar_params[key] = val

        if self._jit_vmap_kernel is None:
            field_axes = {k: 0 for k in field_params.keys()}
            scalar_axes = {k: None for k in scalar_params.keys()}
            self._jit_vmap_kernel = jax.jit(jax.vmap(
                self._val_and_jac_single, 
                in_axes=(0, None, 0, 0, 0, 0, field_axes, scalar_axes)
            ))

        R_jax, K_jax = self._jit_vmap_kernel(
            jnp.array(u_loc_all), self.w_jax, self.detJ_jax, self.pos_gp_jax, 
            self.N_jax, self.B_x_jax, field_params, scalar_params
        )
        
        R_loc_signed = np.array(R_jax) * self.dof_signs_all
        K_loc_signed = np.array(K_jax) * (self.dof_signs_all[:, :, None] * self.dof_signs_all[:, None, :])

        n_dofs = self.dof_indices_all.shape[1]
        rows = np.repeat(self.dof_indices_all, n_dofs, axis=1).flatten()
        cols = np.tile(self.dof_indices_all, (1, n_dofs)).flatten()

        R_global = np.zeros(self.V.ndofs)
        np.add.at(R_global, self.dof_indices_all.flatten(), R_loc_signed.flatten())
        K_global = coo_matrix((K_loc_signed.flatten(), (rows, cols)), shape=(self.V.ndofs, self.V.ndofs)).tolil()

        print(f"Assembly took {time.time() - start_time:.3f} seconds")
        return R_global, K_global


class MixedAssembler:
    def __init__(self, mixed_space, weak_form, quad_degree=2):
        self.V = mixed_space
        self.weak_form = weak_form
        self.quad_degree = quad_degree
        sig = inspect.signature(self.weak_form)
        self.expects_state = 'state' in sig.parameters
        self.geom_element = self.V.spaces[0].mesh.geom_element
        
        self._setup_quadrature_cache()
        self._precompute_geometry()
        self._setup_jax_kernel()
        self._jit_vmap_kernel = None

    def _setup_quadrature_cache(self):
        element = self.V.spaces[0].element
        self.pts, self.wgts = element.get_volume_quadrature(self.quad_degree)
        self.N_cache, self.B_cache = [], []
        for V in self.V.spaces:
            self.N_cache.append([V.element.shape_functions(gp) for gp in self.pts])
            self.B_cache.append([V.element.shape_gradients_reference(gp) for gp in self.pts])

    def _precompute_geometry(self):
        n_elems = len(self.V.spaces[0].mesh.cells)
        n_quad = len(self.wgts)
        n_spaces = len(self.V.spaces)

        detJ_all, pos_gp_all = [], []
        N_spaces_all = [[] for _ in range(n_spaces)]
        B_x_spaces_all = [[] for _ in range(n_spaces)]

        self.dof_indices_all, self.dof_signs_all = [], []

        for e in range(n_elems):
            geom_node_indices = self.V.spaces[0].mesh.cells[e][1]
            elem_nodes = self.V.spaces[0].mesh.points[geom_node_indices]

            e_dofs, e_signs = [], []
            for i, V in enumerate(self.V.spaces):
                e_dofs.append(V.get_dofs(e) + self.V.offsets[i])
                e_signs.append(V.get_dof_signs(e))
            self.dof_indices_all.append(np.concatenate(e_dofs))
            self.dof_signs_all.append(np.concatenate(e_signs))

            detJ_e, pos_gp_e = [], []
            N_e_spaces = [[] for _ in range(n_spaces)]
            B_x_e_spaces = [[] for _ in range(n_spaces)]

            for k in range(n_quad):
                gp = self.pts[k]
                B_ref_geom = self.geom_element.shape_gradients_reference(gp)
                J = self.geom_element.jacobian(elem_nodes, B_ref=B_ref_geom)
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                N_geom = self.geom_element.shape_functions(gp)
                pos_gp_e.append(N_geom @ elem_nodes)
                detJ_e.append(detJ)

                for i, V in enumerate(self.V.spaces):
                    N_e_spaces[i].append(V.element.push_forward_values(self.N_cache[i][k], J, detJ))
                    B_x_e_spaces[i].append(V.element.push_forward_derivatives(self.B_cache[i][k], J, invJ, detJ))

            detJ_all.append(detJ_e)
            pos_gp_all.append(pos_gp_e)
            for i in range(n_spaces):
                N_spaces_all[i].append(N_e_spaces[i])
                B_x_spaces_all[i].append(B_x_e_spaces[i])

        self.detJ_jax = jnp.array(detJ_all)
        self.pos_gp_jax = jnp.array(pos_gp_all)
        self.w_jax = jnp.array(self.wgts)
        
        self.N_jax_spaces = tuple([jnp.array(N_spaces_all[i]) for i in range(n_spaces)])
        self.B_x_jax_spaces = tuple([jnp.array(B_x_spaces_all[i]) for i in range(n_spaces)])

        self.dof_indices_all = np.array(self.dof_indices_all)
        self.dof_signs_all = np.array(self.dof_signs_all)

    def _setup_jax_kernel(self):
        n_quad = len(self.wgts)
        n_spaces = len(self.V.spaces)
        sizes = [len(V.cell_dofs[0]) * V.n_components for V in self.V.spaces]
        mapping_types = [V.element.mapping_type for V in self.V.spaces]
        n_components_list = [V.n_components for V in self.V.spaces]

        def _core_residual_single(U_loc_flat, w_arr, detJ_arr, pos_gp_arr, N_spaces, B_x_spaces, field_params, single_field_params, scalar_params):
            locals_split = []
            cursor = 0
            for i in range(n_spaces):
                size = sizes[i]
                chunk = U_loc_flat[cursor:cursor+size]
                locals_split.append(chunk.reshape((-1, n_components_list[i])))
                cursor += size

            fields_split = {}
            for key, val in field_params.items():
                f_split = []
                f_cursor = 0
                for i in range(n_spaces):
                    size = sizes[i]
                    chunk = val[f_cursor:f_cursor+size]
                    f_split.append(chunk.reshape((-1, n_components_list[i])))
                    f_cursor += size
                fields_split[key] = f_split

            R_loc = jnp.zeros_like(U_loc_flat)

            for k in range(n_quad):
                w, detJ, pos_gp = w_arr[k], detJ_arr[k], pos_gp_arr[k]
                mapped = []
                for i in range(n_spaces):
                    N, B_x = N_spaces[i][k], B_x_spaces[i][k]
                    u_loc = locals_split[i]
                    if mapping_types[i] == "Hdiv":
                        u_gp = jnp.dot(N.T, u_loc)
                        grad_u = jnp.dot(B_x, u_loc)
                    else:
                        u_gp = jnp.dot(N, u_loc)
                        grad_u = jnp.dot(B_x, u_loc)
                    mapped.append((N, B_x, u_gp, grad_u))
                
                if self.expects_state:
                    gp_state = {}
                    for key, val in single_field_params.items():
                        N_0, B_x_0 = N_spaces[0][k], B_x_spaces[0][k]
                        gp_state[key] = jnp.dot(N_0, val)
                        gp_state[f'grad_{key}'] = jnp.dot(B_x_0, val)

                    for key, f_split in fields_split.items():
                        mapped_state = []
                        for i in range(n_spaces):
                            N, B_x = N_spaces[i][k], B_x_spaces[i][k]
                            if mapping_types[i] == "Hdiv":
                                u_gp_state = jnp.dot(N.T, f_split[i])
                                grad_u_state = jnp.dot(B_x, f_split[i])
                            else:
                                u_gp_state = jnp.dot(N, f_split[i])
                                grad_u_state = jnp.dot(B_x, f_split[i])
                            mapped_state.append((u_gp_state, grad_u_state))
                        gp_state[key] = mapped_state
                        
                    for key, val in scalar_params.items():
                        gp_state[key] = val
                        
                    contribs = self.weak_form(mapped, pos_gp, 0, gp_state)
                else:
                    contribs = self.weak_form(mapped, pos_gp, 0)

                flat_contribs = jnp.concatenate([c.flatten() for c in contribs])
                R_loc = R_loc + flat_contribs * detJ * w

            return R_loc

        _compute_jac_single = jax.jacfwd(_core_residual_single, argnums=0)

        def val_and_jac_single(U, w, dJ, pos, Ns, Bs, f_p, sf_p, s_p):
            return _core_residual_single(U, w, dJ, pos, Ns, Bs, f_p, sf_p, s_p), _compute_jac_single(U, w, dJ, pos, Ns, Bs, f_p, sf_p, s_p)

        self._val_and_jac_single = val_and_jac_single

    def assemble(self, U_current, global_params=None):
        start_time = time.time()
        if global_params is None: global_params = {}

        U_loc_all = U_current[self.dof_indices_all] * self.dof_signs_all

        field_params, single_field_params, scalar_params = {}, {}, {}
        n_elems = len(self.V.spaces[0].mesh.cells)

        if self.expects_state:
            for key, val in global_params.items():
                if isinstance(val, tuple) and len(val) == 2 and hasattr(val[0], 'get_dofs'):
                    V_field, arr = val
                    f_dofs = np.array([V_field.get_dofs(e) for e in range(n_elems)])
                    f_signs = np.array([V_field.get_dof_signs(e) for e in range(n_elems)])
                    f_n_nodes = len(V_field.cell_dofs[0])
                    sliced = arr[f_dofs] * f_signs
                    single_field_params[key] = sliced.reshape((-1, f_n_nodes, V_field.n_components))
                elif isinstance(val, np.ndarray) and val.size == self.V.ndofs:
                    field_params[key] = val[self.dof_indices_all] * self.dof_signs_all
                else:
                    scalar_params[key] = val

        if self._jit_vmap_kernel is None:
            field_axes = {k: 0 for k in field_params.keys()}
            single_field_axes = {k: 0 for k in single_field_params.keys()}
            scalar_axes = {k: None for k in scalar_params.keys()}
            self._jit_vmap_kernel = jax.jit(jax.vmap(
                self._val_and_jac_single,
                in_axes=(0, None, 0, 0, 0, 0, field_axes, single_field_axes, scalar_axes)
            ))

        R_jax, K_jax = self._jit_vmap_kernel(
            jnp.array(U_loc_all), self.w_jax, self.detJ_jax, self.pos_gp_jax,
            self.N_jax_spaces, self.B_x_jax_spaces, field_params, single_field_params, scalar_params
        )

        R_loc_signed = np.array(R_jax) * self.dof_signs_all
        K_loc_signed = np.array(K_jax) * (self.dof_signs_all[:, :, None] * self.dof_signs_all[:, None, :])

        n_dofs = self.dof_indices_all.shape[1]
        rows = np.repeat(self.dof_indices_all, n_dofs, axis=1).flatten()
        cols = np.tile(self.dof_indices_all, (1, n_dofs)).flatten()

        R_global = np.zeros(self.V.ndofs)
        np.add.at(R_global, self.dof_indices_all.flatten(), R_loc_signed.flatten())
        K_global = coo_matrix((K_loc_signed.flatten(), (rows, cols)), shape=(self.V.ndofs, self.V.ndofs)).tolil()

        print(f"Mixed Assembly took {time.time() - start_time:.3f} seconds")
        return R_global, K_global

def assemble_scalar(function_space, integrand, u_sol=None, quad_degree=2):
    # Setup quadrature and cache
    geom_element = function_space.mesh.geom_element
    pts, wgts = geom_element.get_volume_quadrature(quad_degree)
    
    # Precompute all element geometries and shape functions 
    # Might want to cache these at the mesh level later
    cells = function_space.mesh.cells
    points = function_space.mesh.points
    n_elems = len(cells)
    
    # Extract element nodes array of shape (n_elems, nodes_per_elem, dim)
    elem_nodes = np.array([points[c[1]] for c in cells])
    
    if u_sol is not None:
        # Extract local dofs array of shape (n_elems, local_dofs)
        dofs_all = np.array([function_space.get_dofs(e) for e in range(n_elems)])
        signs_all = np.array([function_space.get_dof_signs(e) for e in range(n_elems)])
        u_loc_all = (u_sol[dofs_all] * signs_all).reshape((n_elems, -1, function_space.n_components))
    else:
        u_loc_all = jnp.zeros((n_elems, 1, 1)) # Dummy shape

    # Extract mapping type, will default to H1 otherwise
    mapping_type = getattr(function_space.element, "mapping_type", "H1")

    def cell_integral(nodes, u_loc, cell_idx):
        cell_val = 0.0
        for k, (gp, w) in enumerate(zip(pts, wgts)):
            B_ref_geom = geom_element.shape_gradients_reference(gp)
            J = geom_element.jacobian(nodes, B_ref=B_ref_geom)
            detJ = jnp.linalg.det(J)
            invJ = jnp.linalg.inv(J)
            
            N_geom = geom_element.shape_functions(gp)
            pos_gp = jnp.dot(N_geom, nodes)
            
            u_gp, grad_u = None, None
            if u_sol is not None:
                N_ref = function_space.element.shape_functions(gp)
                dN_ref = function_space.element.shape_gradients_reference(gp)
                N = function_space.element.push_forward_values(N_ref, J, detJ)
                dN = function_space.element.push_forward_derivatives(dN_ref, J, invJ, detJ)
                
                if mapping_type == "Hdiv":
                    u_gp = jnp.dot(N.T, u_loc)
                else:
                    u_gp = jnp.dot(N, u_loc) 

                grad_u = jnp.dot(dN, u_loc)
                
            f_val = integrand(u_gp, grad_u, pos_gp, cell_idx)
            cell_val += f_val * detJ * w
        return cell_val

    # Vectorize across all elements
    cell_indices = jnp.arange(n_elems)
    vmap_integral = jax.vmap(cell_integral, in_axes=(0, 0, 0))
    
    # Calculate and sum all element contributions
    all_cell_vals = vmap_integral(elem_nodes, u_loc_all, cell_indices)
    return jnp.sum(all_cell_vals)