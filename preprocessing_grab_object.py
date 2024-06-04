import os
import os.path as osp

import pymeshlab as ml
import glob
import tqdm


def main():
    original_folder = 'data/grab/contact_meshes'
    save_folder = 'data/grab/processed_object_meshes'
    os.makedirs(save_folder, exist_ok=True)

    n_target_vertices = 4000

    original_meshes = glob.glob(os.path.join(original_folder, "*.ply"))
    for mesh_path in tqdm.tqdm(original_meshes, desc="Processing object meshes"):
        basename = os.path.basename(mesh_path)
        ms = ml.MeshSet(verbose=0)
        ms.load_new_mesh(mesh_path)
        m = ms.current_mesh()
        TARGET = n_target_vertices
        numFaces = 100 + 2 * TARGET
        while (ms.current_mesh().vertex_number() > TARGET):
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
        m = ms.current_mesh()
        print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
        ms.save_current_mesh(osp.join(save_folder, basename))

if __name__ == "__main__":
    main()