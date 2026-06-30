import numpy as np
from fem_engine.fem_core import gauss_legendre_quadrature

class Element:
    def __init__(self, dim, n_nodes, n_boundaries, mapping_type = "H1"):
        self.dim = dim
        self.n_nodes = n_nodes
        self.n_boundaries = n_boundaries
        self.mapping_type = mapping_type

    def shape_functions(self, xi): raise NotImplementedError
    def shape_gradients_reference(self, xi): raise NotImplementedError
    
    def jacobian(self, nodes, xi=None, B_ref=None):
        if B_ref is None:
            B_ref = self.shape_gradients_reference(xi)
        return nodes.T @ B_ref.T
    
    def get_local_node_indices(self):
        """Return indices of nodes this element uses from the mesh cell."""
        raise NotImplementedError
    
    def get_ref_coords(self):
        """Returns the logical reference coordinates of the element's DOFs."""
        raise NotImplementedError
    
    def get_entity_dof_map(self):
        raise NotImplementedError
    
    def get_facet_dofs(self, facet):
        raise NotImplementedError

    def get_volume_quadrature(self, degree):
        """Returns: flat_list_of_points, flat_list_of_weights"""
        raise NotImplementedError

    def get_boundary_quadrature(self, bnd_idx, degree):
        """Returns: flat_list_of_points, flat_list_of_weights, local_node_indices"""
        raise NotImplementedError
    
    def map_to_reference(self, nodes, x_phys, tol=1e-10, max_iter=10):
        xi = np.array([0.0, 0.0])  # initial guess

        for _ in range(max_iter):
            N = self.shape_functions(xi)
            dN = self.shape_gradients_reference(xi)

            x_current = N @ nodes

            R = x_current - x_phys
            if np.linalg.norm(R) < tol:
                return xi

            J = nodes.T @ dN.T  # dx/dxi , or just self.jacobian(nodes, xi, dN)

            xi -= np.linalg.solve(J, R)

        raise RuntimeError("Mapping did not converge")
    
    def push_forward_values(self, N_ref, J, detJ):
        """Maps reference values to physical space"""
        if self.mapping_type == "H1":
            return N_ref
        elif self.mapping_type == "Hdiv":
            # Contravariant Piola: N_ref is shape (n_dofs, dim)
            # J @ N_ref[i]/ detJ
            return (N_ref @ J.T) / detJ
        else:
            raise ValueError(f"Unknown mapping type: {self.mapping_type}")
        
    def push_forward_derivatives(self, B_ref, J, invJ, detJ):
        """Maps reference derivatives to physical space"""
        if self.mapping_type == "H1":
            return invJ.T @ B_ref
        elif self.mapping_type == "Hdiv":
            # For H(div), B_ref is the reference divergence
            return B_ref / detJ
    
    
    @property
    def local_topology(self):
        """
        Defines the local node indices that constitute the topological entities.
        Returns a dict mapping entity type to a list of node index lists.
        """
        raise NotImplementedError

class Quad4(Element):
    def __init__(self):
        # 2D, 4 nodes, 4 edges
        super().__init__(dim=2, n_nodes=4, n_boundaries=4)
        # 1 DOF per corner, 0 on edges, 0 in center
        self.entity_dofs = {'vertex': 1, 'edge': 0, 'cell': 0}

    def shape_functions(self, xi):
        xi1, xi2 = xi
        return np.array([
            0.25*(1-xi1)*(1-xi2),
            0.25*(1+xi1)*(1-xi2),
            0.25*(1+xi1)*(1+xi2),
            0.25*(1-xi1)*(1+xi2)
        ])

    def shape_gradients_reference(self, xi):
        xi1, xi2 = xi
        return np.array([
            [-(1-xi2),  (1-xi2),  (1+xi2), -(1+xi2)],
            [-(1-xi1), -(1+xi1),  (1+xi1),  (1-xi1)]
        ]) * 0.25

    def get_volume_quadrature(self, degree):
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        pts, wgts = [], []
        for i in range(degree):
            for j in range(degree):
                pts.append(np.array([xi_1d[i], xi_1d[j]]))
                wgts.append(weights_1d[i] * weights_1d[j])
        return pts, wgts

    def get_boundary_quadrature(self, bnd_idx, degree):
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        mappings = [
            lambda s: np.array([ s, -1.0]),  # Edge 0: Bottom
            lambda s: np.array([ 1.0,  s]),  # Edge 1: Right
            lambda s: np.array([-s,  1.0]),  # Edge 2: Top
            lambda s: np.array([-1.0, -s])   # Edge 3: Left
        ]
        edge_nodes = [[0, 1], [1, 2], [2, 3], [3, 0]]
        pts = [mappings[bnd_idx](s) for s in xi_1d]
        return pts, weights_1d, edge_nodes[bnd_idx]
    
    def get_local_node_indices(self):
        return [0, 1, 2, 3]
    
    def get_ref_coords(self):
        return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    
    def get_entity_dof_map(self):
        return {
            ('vertex', 0): [0],
            ('vertex', 1): [1],
            ('vertex', 2): [2],
            ('vertex', 3): [3]
        }
    
    def get_facet_dofs(self, facet):
            return [
                [0,1],
                [1,2],
                [2,3],
                [3,0]
            ][facet]
    
    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0, 1], [1, 2], [2, 3], [3, 0]],
            'cell': [[0, 1, 2, 3]]
        }


class Quad9(Element):
    def __init__(self):
        # 2D, 9 nodes, 4 edges
        super().__init__(dim=2, n_nodes=9, n_boundaries=4)
        # 1 DOF per corner, 1 per edge midpoint, 1 in center
        self.entity_dofs = {'vertex': 1, 'edge': 1, 'cell': 1}

    def shape_functions(self, xi):
        x, y = xi
        return np.array([
            0.25 * x*(x-1) * y*(y-1), # 0: bottom-left
            0.25 * x*(x+1) * y*(y-1), # 1: bottom-right
            0.25 * x*(x+1) * y*(y+1), # 2: top-right
            0.25 * x*(x-1) * y*(y+1), # 3: top-left
            0.5 * (1-x**2) * y*(y-1), # 4: bottom-mid
            0.5 * x*(x+1) * (1-y**2), # 5: right-mid
            0.5 * (1-x**2) * y*(y+1), # 6: top-mid
            0.5 * x*(x-1) * (1-y**2), # 7: left-mid
            (1-x**2) * (1-y**2)       # 8: center
        ])

    def shape_gradients_reference(self, xi):
        x, y = xi
        dN_dx = np.array([
            0.25 * (2*x - 1) * y*(y - 1),
            0.25 * (2*x + 1) * y*(y - 1),
            0.25 * (2*x + 1) * y*(y + 1),
            0.25 * (2*x - 1) * y*(y + 1),
            -x * y*(y - 1),
            0.5 * (2*x + 1) * (1 - y**2),
            -x * y*(y + 1),
            0.5 * (2*x - 1) * (1 - y**2),
            -2*x * (1 - y**2)
        ])
        dN_dy = np.array([
            0.25 * x*(x - 1) * (2*y - 1),
            0.25 * x*(x + 1) * (2*y - 1),
            0.25 * x*(x + 1) * (2*y + 1),
            0.25 * x*(x - 1) * (2*y + 1),
            0.5 * (1 - x**2) * (2*y - 1),
            -y * x*(x + 1),
            0.5 * (1 - x**2) * (2*y + 1),
            -y * x*(x - 1),
            -2*y * (1 - x**2)
        ])
        return np.vstack((dN_dx, dN_dy))

    def get_volume_quadrature(self, degree):
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        pts, wgts = [], []
        for i in range(degree):
            for j in range(degree):
                pts.append(np.array([xi_1d[i], xi_1d[j]]))
                wgts.append(weights_1d[i] * weights_1d[j])
        return pts, wgts

    def get_boundary_quadrature(self, bnd_idx, degree):
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        mappings = [
            lambda s: np.array([ s, -1.0]),  # Edge 0: Bottom
            lambda s: np.array([ 1.0,  s]),  # Edge 1: Right
            lambda s: np.array([-s,  1.0]),  # Edge 2: Top
            lambda s: np.array([-1.0, -s])   # Edge 3: Left
        ]
        edge_nodes = [[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]
        pts = [mappings[bnd_idx](s) for s in xi_1d]
        return pts, weights_1d, edge_nodes[bnd_idx]
    
    def get_local_node_indices(self):
        return list(range(9))
    
    def get_ref_coords(self):
        return np.array([
            [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], # Corners
            [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],   # Edge Mids
            [0.0, 0.0]                                          # Center
        ])
    
    def get_entity_dof_map(self):
        return {
            ('vertex', 0): [0],
            ('vertex', 1): [1],
            ('vertex', 2): [2],
            ('vertex', 3): [3],

            ('edge', 0): [4],
            ('edge', 1): [5],
            ('edge', 2): [6],
            ('edge', 3): [7],

            ('cell', 0): [8]
        }

    def get_facet_dofs(self, facet):
        return [
            [0,4,1],
            [1,5,2],
            [2,6,3],
            [3,7,0]
        ][facet]
    
    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0,1], [1,2], [2,3], [3,0]],
            'cell': [[0,1,2,3]]
        }
    
class QuadRT0(Element):
    def __init__(self):
        # 2D, 4 DOFs (1 per edge), H(div) mapping
        super().__init__(dim=2, n_nodes=4, n_boundaries=4, mapping_type="Hdiv")
        # DOFs live purely on edges. 0 on vertices, 0 in cells.
        self.entity_dofs = {'vertex': 0, 'edge': 1, 'cell': 0}

    def shape_functions(self, xi):
        x, y = xi
        # Reference RT0 vectors on [-1, 1]^2
        return np.array([
            [0.0, 0.25 * (y - 1)], # Edge 0 (Bottom)
            [0.25 * (x + 1), 0.0], # Edge 1 (Right)
            [0.0, 0.25 * (y + 1)], # Edge 2 (Top)
            [0.25 * (x - 1), 0.0]  # Edge 3 (Left)
        ])

    def shape_gradients_reference(self, xi):
        # For Hdiv, this returns the reference divergence
        # div(N) = dNx/dx + dNy/dy = 0.25 for all edges
        return np.array([0.25, 0.25, 0.25, 0.25])

    def get_volume_quadrature(self, degree):
        # Re-use standard quad integration
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        pts, wgts = [], []
        for i in range(degree):
            for j in range(degree):
                pts.append(np.array([xi_1d[i], xi_1d[j]]))
                wgts.append(weights_1d[i] * weights_1d[j])
        return pts, wgts

    def get_boundary_quadrature(self, bnd_idx, degree):
        xi_1d, weights_1d = gauss_legendre_quadrature(degree)
        mappings = [
            lambda s: np.array([ s, -1.0]),  # Edge 0: Bottom
            lambda s: np.array([ 1.0,  s]),  # Edge 1: Right
            lambda s: np.array([-s,  1.0]),  # Edge 2: Top
            lambda s: np.array([-1.0, -s])   # Edge 3: Left
        ]
        # Only 1 DOF per edge for RT0
        edge_dofs = [[0], [1], [2], [3]]
        pts = [mappings[bnd_idx](s) for s in xi_1d]
        return pts, weights_1d, edge_dofs[bnd_idx]

    def get_local_node_indices(self):
        # Indices of the 4 DOFs
        return [0, 1, 2, 3]

    def get_ref_coords(self):
        # Logical locations of the DOFs (the edge midpoints)
        return np.array([
            [0.0, -1.0],  # Edge 0 (Bottom)
            [1.0, 0.0],   # Edge 1 (Right)
            [0.0, 1.0],   # Edge 2 (Top)
            [-1.0, 0.0]   # Edge 3 (Left)
        ])

    def get_entity_dof_map(self):
        return {
            ('edge', 0): [0],
            ('edge', 1): [1],
            ('edge', 2): [2],
            ('edge', 3): [3]
        }

    def get_facet_dofs(self, facet):
        # Which DOF belongs to which facet (boundary edge)
        return [[0], [1], [2], [3]][facet]

    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0,1], [1,2], [2,3], [3,0]],
            'cell': [[0,1,2,3]]
        }
    
class Quad0(Element):
    def __init__(self):
        super().__init__(dim=2, n_nodes=1, n_boundaries=4, mapping_type="H1")
        self.entity_dofs = {'vertex': 0, 'edge': 0, 'cell': 1}

    def shape_functions(self, xi): return np.array([1.0])
    def shape_gradients_reference(self, xi): 
        # Must be shape (dim, n_nodes) -> (2, 1)
        return np.array([[0.0], [0.0]])
    
    def get_volume_quadrature(self, degree):
        return Quad4().get_volume_quadrature(degree) # Steal quad points from Quad4

    def get_boundary_quadrature(self, bnd_idx, degree):
        return Quad4().get_boundary_quadrature(bnd_idx, degree) # Dummy fallback

    def get_local_node_indices(self): return [0]
    def get_ref_coords(self): return np.array([[0.0, 0.0]])
    
    def get_entity_dof_map(self):
        return {('cell', 0): [0]}
        
    def get_facet_dofs(self, facet): return []
    
    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0,1], [1,2], [2,3], [3,0]],
            'cell': [[0,1,2,3]]
        }
    '''
    # The local_topology is actually the same for all Quads... all Quads, in ref space, have 4 vertex, 4 edges and ONE face in 2D
    # One would think that for Quad0 it could be as follows, but actually get_entity_dof_map and entity_dofs do the heavy work in mesh and assembly later...
    @property
    def local_topology(self):
        return {'vertex': [], 'edge': [], 'cell': [[0, 1, 2, 3]]}
    '''
    
class QuadRT1(Element):
    """
    QRT1 element on [-1,1]^2
    Space: P_{2,1} x P_{1,2}
    DOFs: 12 = 8 edge + 4 interior
    """
    def __init__(self):
        super().__init__(dim=2, n_nodes=12, n_boundaries=4, mapping_type="Hdiv")

        self.entity_dofs = {
            'vertex': 0,
            'edge': 2,   # 2 DOFs per edge
            'cell': 4    # 4 interior DOFs total
        }

    def shape_functions(self, xi):
        x, y = xi
        
        # 1D Basis Building Blocks
        L0x, L1x = 0.5 * (1 - x), 0.5 * (1 + x)
        L0y, L1y = 0.5 * (1 - y), 0.5 * (1 + y)
        Bx, By   = 1 - x**2, 1 - y**2

        N = np.zeros((12, 2))

        # EDGE DOFs (Linear flux moments on the edges)
        
        # Edge 0: Bottom (y=-1). Nodes [0, 1] -> x goes from -1 to 1.
        N[0] = [0, -L0x * L0y]  # Peaks at x=-1 (Node 0)
        N[1] = [0, -L1x * L0y]  # Peaks at x=1  (Node 1)

        # Edge 1: Right (x=1). Nodes [1, 2] -> y goes from -1 to 1.
        N[2] = [L1x * L0y, 0]   # Peaks at y=-1 (Node 1)
        N[3] = [L1x * L1y, 0]   # Peaks at y=1  (Node 2)

        # Edge 2: Top (y=1). Nodes [2, 3] -> x goes from 1 to -1.
        N[4] = [0, L1x * L1y]   # Peaks at x=1  (Node 2)
        N[5] = [0, L0x * L1y]   # Peaks at x=-1 (Node 3)

        # Edge 3: Left (x=-1). Nodes [3, 0] -> y goes from 1 to -1.
        N[6] = [-L0x * L1y, 0]  # Peaks at y=1  (Node 3)
        N[7] = [-L0x * L0y, 0]  # Peaks at y=-1 (Node 0)

        # INTERIOR DOFs (Bubble completion for P_{2,1} x P_{1,2})
        N[8]  = [Bx * L0y, 0]
        N[9]  = [Bx * L1y, 0]
        N[10] = [0, L0x * By]
        N[11] = [0, L1x * By]

        return N

    def shape_gradients_reference(self, xi):
        x, y = xi
        div = np.zeros(12)

        # Divergences: div(v) = dv_x/dx + dv_y/dy
        
        # Edge 0
        div[0] = 0.25 * (1 - x)
        div[1] = 0.25 * (1 + x)

        # Edge 1
        div[2] = 0.25 * (1 - y)
        div[3] = 0.25 * (1 + y)

        # Edge 2
        div[4] = 0.25 * (1 + x)
        div[5] = 0.25 * (1 - x)

        # Edge 3
        div[6] = 0.25 * (1 + y)
        div[7] = 0.25 * (1 - y)

        # Interior DOFs
        div[8]  = -x * (1 - y)
        div[9]  = -x * (1 + y)
        div[10] = -y * (1 - x)
        div[11] = -y * (1 + x)

        return div

    def get_volume_quadrature(self, degree):
        xi_1d, w_1d = gauss_legendre_quadrature(degree)
        pts, wgts = [], []
        for i in range(degree):
            for j in range(degree):
                pts.append(np.array([xi_1d[i], xi_1d[j]]))
                wgts.append(w_1d[i] * w_1d[j])
        return pts, wgts

    def get_boundary_quadrature(self, bnd_idx, degree):
        xi_1d, w_1d = gauss_legendre_quadrature(degree)

        mappings = [
            lambda s: np.array([ s, -1.0]),  # bottom
            lambda s: np.array([ 1.0,  s]),  # right
            lambda s: np.array([-s,  1.0]),  # top
            lambda s: np.array([-1.0, -s])   # left
        ]

        edge_dofs = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7]
        ]

        pts = [mappings[bnd_idx](s) for s in xi_1d]
        return pts, w_1d, edge_dofs[bnd_idx]

    def get_local_node_indices(self):
        return list(range(12))

    def get_ref_coords(self):
        return np.zeros((12, 2))

    def get_entity_dof_map(self):
        return {
            ('edge', 0): [0, 1],
            ('edge', 1): [2, 3],
            ('edge', 2): [4, 5],
            ('edge', 3): [6, 7],
            ('cell', 0): [8, 9, 10, 11]
        }

    def get_facet_dofs(self, facet):
        return [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7]
        ][facet]
    
    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0,1], [1,2], [2,3], [3,0]],
            'cell': [[0,1,2,3]]
        }
    
class Quad1Dc(Element):
    """
    Discontinuous Bilinear Element (Q1_dc).
    4 DOFs, all assigned to the interior of the cell so they are not shared.
    Used for the pressure space when paired with RT1.
    """
    def __init__(self):
        super().__init__(dim=2, n_nodes=4, n_boundaries=4, mapping_type="H1")
        # 0 DOFs on vertices/edges, 4 DOFs in the interior
        self.entity_dofs = {'vertex': 0, 'edge': 0, 'cell': 4}

    def shape_functions(self, xi):
        x, y = xi
        return np.array([
            0.25 * (1 - x) * (1 - y),
            0.25 * (1 + x) * (1 - y),
            0.25 * (1 + x) * (1 + y),
            0.25 * (1 - x) * (1 + y)
        ])

    def shape_gradients_reference(self, xi):
        x, y = xi
        return np.array([
            [-0.25 * (1 - y),  0.25 * (1 - y),  0.25 * (1 + y), -0.25 * (1 + y)],
            [-0.25 * (1 - x), -0.25 * (1 + x),  0.25 * (1 + x),  0.25 * (1 - x)]
        ])

    def get_volume_quadrature(self, degree):
        return Quad4().get_volume_quadrature(degree)

    def get_boundary_quadrature(self, bnd_idx, degree):
        return Quad4().get_boundary_quadrature(bnd_idx, degree)

    def get_local_node_indices(self):
        return [0, 1, 2, 3]

    def get_ref_coords(self):
        return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    def get_entity_dof_map(self):
        return {
            ('cell', 0): [0, 1, 2, 3]
        }

    def get_facet_dofs(self, facet):
        return []

    # THIS HERE IS THE SAME FOR ANY QUAD
    @property
    def local_topology(self):
        return {
            'vertex': [[0], [1], [2], [3]],
            'edge': [[0,1], [1,2], [2,3], [3,0]],
            'cell': [[0,1,2,3]]
        }