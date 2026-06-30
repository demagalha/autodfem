import numpy as np
import meshio
from fem_engine.mesh import Mesh

def import_mesh(filepath, element_type="quad"):
    """
    Imports an external mesh file and converts it into Mesh object.
    
    Parameters:
    - filepath: Path to the mesh file (e.g., 'my_geometry.msh' or 'mesh.vtu')
    - element_type: 'quad' for Quad4, 'quad9' for Quad9.
    """
    print(f"Reading external mesh: {filepath}...")
    mesh_data = meshio.read(filepath)

    # 1. Extract and Format Points
    points = mesh_data.points
    # Gmsh and VTK often save 2D meshes in 3D space with Z=0. 
    # Slice off the Z-column so it gets purely 2D coordinates.
    if points.shape[1] == 3 and np.allclose(points[:, 2], 0.0):
        points = points[:, :2]

    # 2. Extract Cells (Connectivity)
    cell_connectivity = None
    
    # meshio stores elements in blocks
    for block in mesh_data.cells:
        if block.type == element_type:
            cell_connectivity = block.data
            break

    if cell_connectivity is None:
        available_types = [block.type for block in mesh_data.cells]
        raise ValueError(
            f"No '{element_type}' elements found in {filepath}! "
            f"Found element types: {available_types}"
        )

    # Format Cells
    # Mesh class expects: [("quad", [n0, n1, n2, n3]), ...]
    formatted_cells = [
        (element_type, list(node_indices)) 
        for node_indices in cell_connectivity
    ]

    print(f"Successfully loaded {len(points)} nodes and {len(formatted_cells)} {element_type} elements.")

    # Return the Mesh
    return Mesh(points, formatted_cells, cell_type=element_type)