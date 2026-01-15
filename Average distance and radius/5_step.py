#For each atom, find the nearest Ni atom and compute the distance.
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def find_closest_distances(atom_positions, ni_positions):
    tree = KDTree(ni_positions)
    distances = tree.query(atom_positions)[0]
    return distances.mean()

def process_lammpstrj(file_3_step, file_ni, output_excel):

    with open(file_3_step, 'r') as file_3, open(file_ni, 'r') as file_ni:
        block_3 = []
        block_ni = []
        ni_blocks = {}
        results = []

        for line in file_ni:
            if line.startswith("ITEM: TIMESTEP"):
                if block_ni:
                    timestep = block_ni[1].strip()
                    ni_blocks[timestep] = block_ni[:]
                    block_ni = []
                block_ni.append(line)
            else:
                block_ni.append(line)
        if block_ni:
            timestep = block_ni[1].strip()
            ni_blocks[timestep] = block_ni[:]

        for line in file_3:
            if line.startswith("ITEM: TIMESTEP"):
                if block_3:
                    timestep = block_3[1].strip()
                    if timestep in ni_blocks:
                        process_block(block_3, ni_blocks[timestep], results)
                    block_3 = []
                block_3.append(line)
            else:
                block_3.append(line)

        if block_3:
            timestep = block_3[1].strip()
            if timestep in ni_blocks:
                process_block(block_3, ni_blocks[timestep], results)

    df = pd.DataFrame(results, columns=["TIMESTEP", "Average_Distance"])
    df.to_excel(output_excel, index=False)
    print(f"Processing complete. The results have been saved to {output_excel}")

def process_block(block_3, block_ni, results):

    atom_positions_3 = []
    ni_positions = []

    atom_lines = block_3[9:]
    for atom_line in atom_lines:
        cols = atom_line.split()
        if len(cols) >= 9:
            x, y, z = map(float, cols[6:9])
            atom_positions_3.append([x, y, z])

    #Skip blocks with fewer than 15 atoms.
    if len(atom_positions_3) <= 15:
        return

    atom_lines_ni = block_ni[9:]
    for atom_line in atom_lines_ni:
        cols = atom_line.split()
        if len(cols) >= 2 and cols[1] == '1':
            x, y, z = map(float, cols[6:9])
            ni_positions.append([x, y, z])

    if len(ni_positions) == 0:
        return

    atom_positions_3 = np.array(atom_positions_3)
    ni_positions = np.array(ni_positions)

    #Calculate the average distance.
    avg_distance = find_closest_distances(atom_positions_3, ni_positions)
    timestep = block_3[1].strip()
    results.append([timestep, avg_distance])

file_3_step_path = '3_step.lammpstrj'
file_ni_path = 'ni.lammpstrj'
output_excel_path = 'average_distance_results.xlsx'

process_lammpstrj(file_3_step_path, file_ni_path, output_excel_path)
