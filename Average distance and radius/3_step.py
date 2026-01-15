#Obtain the largest carbon cluster.
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

def find_largest_cluster(atoms, cutoff=1.8):
    if len(atoms) == 0:
        return []

    tree = KDTree(atoms)
    visited = set()
    clusters = []

    for i in range(len(atoms)):
        if i not in visited:
            cluster = []
            stack = [i]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    neighbors = tree.query_ball_point(atoms[current], cutoff)
                    stack.extend(neighbors)
            clusters.append(cluster)

    largest_cluster = max(clusters, key=len)
    return largest_cluster

def process_lammpstrj(input_file, output_file):

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        block = []
        for line in infile:
            if line.startswith("ITEM: TIMESTEP"):
                if block:

                    process_block(block, outfile)

                block = [line]
            else:
                block.append(line)

        if block:
            process_block(block, outfile)

    print(f"Processing complete. The results have been saved to {output_file}")

def process_block(block, outfile):

    header = block[:9]
    atom_lines = block[9:]
    atom_positions = []
    atom_data = []

    for atom_line in atom_lines:
        cols = atom_line.split()
        if len(cols) >= 9:
            x, y, z = map(float, cols[6:9])
            atom_positions.append([x, y, z])
            atom_data.append(atom_line.strip())

    atom_positions = np.array(atom_positions)

    largest_cluster_indices = find_largest_cluster(atom_positions)

    filtered_atoms = [atom_data[i] for i in largest_cluster_indices]

    new_atom_count = len(filtered_atoms)
    header[3] = f"{new_atom_count}\n"
    
    outfile.writelines(header)
    outfile.writelines(f"{atom}\n" for atom in filtered_atoms)

process_lammpstrj('2_step.lammpstrj', '3_step.lammpstrj')
