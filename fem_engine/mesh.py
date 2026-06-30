import numpy as np
import fem_engine.element as el

# Some of what is written is done with the assumption of quads...
# The boundary box and finding (used for the evaluate_field_at_point) should not be here... but for now...

class Mesh:
    def __init__(self, points, cells, cell_type="quad"):
        self.points = np.asarray(points)
        self.cells = cells
        self.cell_type = cell_type
        self.n_nodes = len(points)
        # Automatically infer spatial dimension from points array, maybe not?
        self.dim = self.points.shape[1]
        self.element_bboxes = self._compute_element_bboxes()
        self.build_bvh()

        if cell_type == "quad":
            self.geom_element = el.Quad4()
        elif cell_type == "quad9":
            self.geom_element = el.Quad9()
        else:
            raise NotImplementedError(f"Unsupported geometry type: {cell_type}")
    
    def _compute_element_bboxes(self):
        bboxes = []

        for cell in self.cells:
            node_ids = cell[1]
            pts = self.points[node_ids]

            xmin = np.min(pts, axis=0)
            xmax = np.max(pts, axis=0)

            bboxes.append((xmin, xmax))

        return bboxes
    
    def build_bvh(self, max_leaf_size=8):

        def bbox_union(bboxes):
            xmin = np.min([b[0] for b in bboxes], axis=0)
            xmax = np.max([b[1] for b in bboxes], axis=0)
            return xmin, xmax

        def build(indices):
            if len(indices) <= max_leaf_size:
                bbox = bbox_union([self.element_bboxes[i] for i in indices])
                return BVHNode(bbox, element_indices=indices)

            # compute split axis (largest extent)
            bboxes = [self.element_bboxes[i] for i in indices]
            xmin = np.min([b[0] for b in bboxes], axis=0)
            xmax = np.max([b[1] for b in bboxes], axis=0)
            extent = xmax - xmin
            axis = np.argmax(extent)

            # sort by centroid
            centroids = [
                0.5 * (self.element_bboxes[i][0] + self.element_bboxes[i][1])
                for i in indices
            ]
            centroids = np.array(centroids)

            sorted_idx = np.argsort(centroids[:, axis])
            indices = [indices[i] for i in sorted_idx]

            mid = len(indices) // 2
            left = build(indices[:mid])
            right = build(indices[mid:])

            bbox = bbox_union([left.bbox, right.bbox])
            return BVHNode(bbox, left, right)

        self.bvh_root = build(list(range(len(self.cells))))

    def find_candidates(self, x):
        def recurse(node):
            xmin, xmax = node.bbox

            if np.any(x < xmin) or np.any(x > xmax):
                return []

            if node.is_leaf:
                return node.element_indices

            return recurse(node.left) + recurse(node.right)

        return recurse(self.bvh_root)

class FunctionSpace:
    def __init__(self, mesh, element, n_components=1):
        self.mesh = mesh
        self.element = element
        self.n_components = n_components
        
        # vertex_to_base_dof maps a mesh vertex ID to its assigned base DOF ID
        # This is CRITICAL for post-processing and plotting.
        # it WAS critical before... but i'll remove it later probably
        self.vertex_to_base_dof = {}
        
        # Build the pure, contiguous topological DOF map
        self.cell_dofs, self.cell_dof_signs, self.total_base_dofs = self._build_dofmap()
        self.ndofs = self.total_base_dofs * self.n_components

    def _build_dofmap(self):
        cell_dofs = []
        cell_dof_signs = [] # either +1.0 or -1.0
        if len(self.mesh.cells) == 0:
            return [], [], 0

        current_dof = 0
        entity_to_dof = {} # Maps ('type', ID) -> DOF index
        
        # Grab the element's blueprint for where DOFs belong
        local_dof_map = self.element.get_entity_dof_map()
        local_topology = self.element.local_topology

        for e, cell in enumerate(self.mesh.cells):
            global_nodes = cell[1]
            
            # Pre-allocate the local-to-global map for this specific cell
            # The length matches the total number of nodes/DOFs in the element
            cell_dof_array = [0] * self.element.n_nodes

            cell_sign_array = [1.0] * self.element.n_nodes # default to +1.0
            
            # 1. Vertex DOFs
            if self.element.entity_dofs.get('vertex', 0) > 0:
                for local_idx, topo in enumerate(local_topology['vertex']):
                    global_v = global_nodes[topo[0]]
                    entity_key = ('vertex', global_v)
                    
                    if entity_key not in entity_to_dof:
                        entity_to_dof[entity_key] = current_dof
                        self.vertex_to_base_dof[global_v] = current_dof
                        current_dof += self.element.entity_dofs['vertex']
                    
                    # Slot the assigned global DOFs into the correct local array positions
                    base_assigned_dof = entity_to_dof[entity_key]
                    for offset, local_dof_idx in enumerate(local_dof_map[('vertex', local_idx)]):
                        cell_dof_array[local_dof_idx] = base_assigned_dof + offset
                        
            # 2. Edge DOFs
            if self.element.entity_dofs.get('edge', 0) > 0:
                for local_idx, topo in enumerate(local_topology['edge']):
                    # Sort the global corner IDs to create a unique, hashable key for the edge
                    # NOTE: 'topo' must strictly contain only the 2 corner nodes of the edge
                    # to ensure the sorted global key matches across adjacent elements.
                    # FOR QUADS TOPO IS THE SAME!@@##@
                    # IF SOMETHING BREAKS WHEN AND IF I ADD TRIG LATER WILL BE BECAUSE OF THIS
                    global_edge_verts = tuple(sorted([global_nodes[v] for v in topo]))
                    entity_key = ('edge', global_edge_verts)

                    # Extract the global IDs of the two corners making up this edge
                    v0, v1 = global_nodes[topo[0]], global_nodes[topo[1]]
                    
                    if entity_key not in entity_to_dof:
                        entity_to_dof[entity_key] = current_dof
                        current_dof += self.element.entity_dofs['edge']
                        
                    base_assigned_dof = entity_to_dof[entity_key]

                    # Orientation check
                    # Main idea: if local node order (v0->v1) opposes global order (min->max)
                    # AND the element uses Hdiv mapping, must flip the sign
                    is_flipped = v0 > v1
                    sign = -1.0 if (self.element.mapping_type == "Hdiv" and is_flipped) else 1.0

                    dofs_on_edge = local_dof_map[('edge', local_idx)]
                    for offset, local_dof_idx in enumerate(dofs_on_edge):
                        actual_offset = offset
                        
                        if self.element.mapping_type == "Hdiv" and is_flipped:
                            # Swap 0 with 1 (or reverse however many DOFs there are)
                            actual_offset = (len(dofs_on_edge) - 1) - offset
                            
                        cell_dof_array[local_dof_idx] = base_assigned_dof + actual_offset
                        cell_sign_array[local_dof_idx] = sign

            # 3. Cell (Bubble) DOFs
            if self.element.entity_dofs.get('cell', 0) > 0:
                entity_key = ('cell', e)
                
                if entity_key not in entity_to_dof:
                    entity_to_dof[entity_key] = current_dof
                    current_dof += self.element.entity_dofs['cell']
                    
                base_assigned_dof = entity_to_dof[entity_key]
                # Assuming index 0 for the single cell entity
                '''
                Cell: A quadrilateral is just one single polygon. It only has 1 interior. Since counting starts at zero, that single interior is always index 0.

                ('cell', 0)

                There is no ('cell', 1) because a single quadrilateral doesn't have a second, separate interior volume hidden inside it.

                Even if we shove 100 bubble DOFs into that element (like in a high-order hierarchical element), they all live inside that exact same, single interior volume. Therefore, they all belong to entity ('cell', 0).

                The hardcoded 0 just means: "Give me the DOFs mapped to the first (and only) interior volume of this element."
                '''
                for offset, local_dof_idx in enumerate(local_dof_map[('cell', 0)]):
                    cell_dof_array[local_dof_idx] = base_assigned_dof + offset
                    
            cell_dofs.append(cell_dof_array)
            cell_dof_signs.append(cell_sign_array)
            
        return cell_dofs, cell_dof_signs, current_dof
    
    def get_dof_signs(self, cell_index, component=None):
        base_signs = self.cell_dof_signs[cell_index]
        if self.n_components == 1:
            return np.array(base_signs)
        if component is not None:
            return np.array(base_signs)
        signs = []
        for s in base_signs:
            signs.extend([s] * self.n_components)
        return np.array(signs)

    def get_dofs(self, cell_index, component=None):
        base_dofs = self.cell_dofs[cell_index]
        
        if self.n_components == 1:
            return np.array(base_dofs)
        
        if component is not None:
            return np.array([dof * self.n_components + component for dof in base_dofs])
            
        dofs = []
        for dof in base_dofs:
            dofs.extend([dof * self.n_components + c for c in range(self.n_components)])
        return np.array(dofs)
    
class MixedFunctionSpace:
    def __init__(self, spaces):
        self.spaces = spaces
        
        self.offsets = []
        offset = 0
        for V in spaces:
            self.offsets.append(offset)
            offset += V.ndofs
        
        self.ndofs = offset

    def split(self, U):
        fields = []
        for V, offset in zip(self.spaces, self.offsets):
            fields.append(U[offset:offset + V.ndofs])
        return fields

def create_rectangle_mesh(L, W, n_x, n_y, x0=-0.5, y0=-0.5):
    """Factory function to generate a 2D quadrilateral mesh."""
    x = np.linspace(x0, L, n_x)
    y = np.linspace(y0, W, n_y)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])

    cells = [("quad", [i + j * n_x, i + 1 + j * n_x, i + 1 + (j + 1) * n_x, i + (j + 1) * n_x]) 
             for j in range(n_y - 1) for i in range(n_x - 1)]
    
    return Mesh(points, cells, cell_type="quad")

def create_quadratic_rectangle_mesh(L, W, n_x, n_y, x0=-0.5, y0=-0.5):
    """Factory function to generate a 2D 9-node quadrilateral mesh."""
    # A mesh with N quadratic elements needs 2*N + 1 nodes per axis
    n_pts_x = 2 * n_x + 1
    n_pts_y = 2 * n_y + 1
    
    x = np.linspace(x0, L, n_pts_x)
    y = np.linspace(y0, W, n_pts_y)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])

    cells = []
    for j in range(n_y):
        for i in range(n_x):
            # Base node indices for the bottom-left corner of the element
            r = 2 * j
            c = 2 * i
            
            # Corner nodes
            n0 = r * n_pts_x + c
            n1 = r * n_pts_x + (c + 2)
            n2 = (r + 2) * n_pts_x + (c + 2)
            n3 = (r + 2) * n_pts_x + c
            
            # Mid-edge and center nodes
            n4 = r * n_pts_x + (c + 1)       # bottom-mid
            n5 = (r + 1) * n_pts_x + (c + 2) # right-mid
            n6 = (r + 2) * n_pts_x + (c + 1) # top-mid
            n7 = (r + 1) * n_pts_x + c       # left-mid
            n8 = (r + 1) * n_pts_x + (c + 1) # center
            
            # Order MUST match the shape function mappings
            cells.append(("quad9", [n0, n1, n2, n3, n4, n5, n6, n7, n8]))
            
    return Mesh(points, cells, cell_type="quad9")

def create_linear_L_mesh(L, W, n_x, n_y, x0=0.0, y0=0.0, cut_x_ratio=0.5, cut_y_ratio=0.5, cut_corner="ur"):
    """
    Structured L-shaped mesh with arbitrary origin.
    Cleans up unused nodes after cutting out the corner.
    """
    x = np.linspace(x0, x0 + L, n_x)
    y = np.linspace(y0, y0 + W, n_y)

    X, Y = np.meshgrid(x, y)
    raw_points = np.column_stack([X.ravel(), Y.ravel()])

    def node(i, j):
        return j * n_x + i

    cut_x = x0 + cut_x_ratio * L
    cut_y = y0 + cut_y_ratio * W

    raw_cells = []

    for j in range(n_y - 1):
        for i in range(n_x - 1):

            x0c, x1c = x[i], x[i+1]
            y0c, y1c = y[j], y[j+1]

            xc = 0.5 * (x0c + x1c)
            yc = 0.5 * (y0c + y1c)

            if cut_corner == "ur":
                remove = (xc > cut_x and yc > cut_y)
            elif cut_corner == "ul":
                remove = (xc < cut_x and yc > cut_y)
            elif cut_corner == "br":
                remove = (xc > cut_x and yc < cut_y)
            elif cut_corner == "bl":
                remove = (xc < cut_x and yc < cut_y)
            else:
                raise ValueError("Invalid cut_corner")

            if remove:
                continue

            n0 = node(i, j)
            n1 = node(i+1, j)
            n2 = node(i+1, j+1)
            n3 = node(i, j+1)

            raw_cells.append([n0, n1, n2, n3])

    # Node Purging and Remapping
    
    # 1. Get a sorted list of all unique node IDs actually used in the cells
    used_nodes = np.unique(raw_cells)
    
    # 2. Extract only the points that are being used
    points = raw_points[used_nodes]
    
    # 3. Create a dictionary to map the old node ID to the new contiguous ID
    node_map = {old_id: new_id for new_id, old_id in enumerate(used_nodes)}
    
    # 4. Remap the cells using the new contiguous indices
    cells = [("quad", [node_map[n] for n in cell]) for cell in raw_cells]

    return Mesh(points, cells, cell_type="quad")

class BVHNode:
    def __init__(self, bbox, left=None, right=None, element_indices=None):
        self.bbox = bbox # (xmin, xmax)
        self.left = left
        self.right = right
        self.element_indices = element_indices # only for leaves
        self.is_leaf = element_indices is not None