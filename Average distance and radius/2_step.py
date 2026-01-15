#Delete carbon atoms with a coordination number of 0.
import numpy as np
from scipy.spatial import KDTree

def calculate_coordination_numbers(atoms, cutoff=1.8):

    tree = KDTree(atoms)
    coord_numbers = np.zeros(len(atoms), dtype=int)
    for i, atom in enumerate(atoms):
        neighbors = tree.query_ball_point(atom, cutoff)
        coord_numbers[i] = len(neighbors) - 1
    return coord_numbers

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
    coord_numbers = calculate_coordination_numbers(atom_positions)

    filtered_atoms = [atom_data[i] for i in range(len(atom_data)) if coord_numbers[i] > 0]

    new_atom_count = len(filtered_atoms)
    header[3] = f"{new_atom_count}\n"

    outfile.writelines(header)
    outfile.writelines(f"{atom}\n" for atom in filtered_atoms)

process_lammpstrj('1_step.lammpstrj', '2_step.lammpstrj')
