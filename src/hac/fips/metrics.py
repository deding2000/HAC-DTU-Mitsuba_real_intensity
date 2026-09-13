import meshio
import trimesh
import numpy as np


def load_as_trimesh(mshfile):
    m = meshio.read(mshfile)
    return trimesh.Trimesh(vertices=m.points, faces=m.cells_dict["triangle"])


def relative_volume_difference_voxelized(meshname1, meshname2, pitch=10):
    """Either argument may be a mesh file path or an already-loaded trimesh.

    Passing a loaded mesh matters when one side is reused across many calls: the
    file read dominates the runtime for a dense reference shape.
    """

    A = (
        meshname1
        if isinstance(meshname1, trimesh.Trimesh)
        else load_as_trimesh(meshname1)
    )
    B = (
        meshname2
        if isinstance(meshname2, trimesh.Trimesh)
        else load_as_trimesh(meshname2)
    )

    # -------------------------------------------------------
    # 1. Compute shared bounding box min/max
    # -------------------------------------------------------
    min_bound = np.minimum(A.bounds[0], B.bounds[0])
    max_bound = np.maximum(A.bounds[1], B.bounds[1])

    # -------------------------------------------------------
    # 2. Append ghost vertices to BOTH meshes
    #    (no faces = does NOT fill voxels)
    # -------------------------------------------------------
    bbox1 = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    bbox1.apply_translation(min_bound)

    bbox2 = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    bbox2.apply_translation(max_bound)

    # -------------------------------------------------------
    # 3. Add the SAME bounding box to BOTH meshes
    # -------------------------------------------------------
    A_padded = trimesh.util.concatenate([A, bbox1, bbox2])
    B_padded = trimesh.util.concatenate([B, bbox1, bbox2])

    # Now A_padded.bounds == B_padded.bounds

    # -------------------------------------------------------
    # 3. Voxelize — identical grid size guaranteed
    # -------------------------------------------------------
    vA = A_padded.voxelized(pitch, method="subdivide").fill()
    vB = B_padded.voxelized(pitch, method="subdivide").fill()

    filled_A = vA.matrix.astype(bool)
    filled_B = vB.matrix.astype(bool)

    # -------------------------------------------------------
    # 4. Volume difference
    # -------------------------------------------------------
    vol_voxel = pitch**3

    vol_D1 = np.sum(np.logical_and(filled_A, np.logical_not(filled_B))) * vol_voxel
    vol_D2 = np.sum(np.logical_and(filled_B, np.logical_not(filled_A))) * vol_voxel

    vol_AandB = np.sum(np.logical_and(filled_A, filled_B)) * vol_voxel
    vol_AorB = np.sum(np.logical_or(filled_A, filled_B)) * vol_voxel

    vol_A = np.sum(filled_A) * vol_voxel
    vol_B = np.sum(filled_B) * vol_voxel

    measure1 = 1 - vol_AandB / vol_AorB
    measure2 = (vol_D1 + vol_D2) / (vol_A + vol_B)

    return measure1, measure2
