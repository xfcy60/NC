#Fit the point cloud data using the least-squares method to determine the sphere center and radius.
import numpy as np
import pandas as pd


def fit_sphere_least_squares(points):

    A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
    b = np.sum(points ** 2, axis=1).reshape(-1, 1)

    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = coeffs[:3].flatten()
    radius = np.sqrt(coeffs[3][0] + np.sum(center ** 2))

    return center, radius


def process_lammpstrj(input_file, output_excel):

    with open(input_file, 'r') as infile:
        block = []
        unit_count = 0
        results = []

        for line in infile:
            if line.startswith("ITEM: TIMESTEP"):
                if block:
                    unit_count += 1
                    process_block(block, results)
                block = [line]
            else:
                block.append(line)

        if block:
            unit_count += 1
            process_block(block, results)

    df = pd.DataFrame(results, columns=["TIMESTEP", "Center", "Radius"])
    df.to_excel(output_excel, index=False)
    print(f"rocessing complete. The results have been saved to {output_excel}")


def process_block(block, results):

    if len(block) < 10:
        return

    timestep = block[1].strip()
    atom_lines = block[9:]
    atom_positions = []

    for atom_line in atom_lines:
        cols = atom_line.split()
        if len(cols) >= 9:
            x, y, z = map(float, cols[6:9])
            atom_positions.append([x, y, z])

    atom_positions = np.array(atom_positions)

    if len(atom_positions) > 15:
        center, radius = fit_sphere_least_squares(atom_positions)
        results.append([timestep, center.tolist(), radius])

input_file_path = '3_step.lammpstrj'
output_excel_path = 'sphere_fit_results.xlsx'

process_lammpstrj(input_file_path, output_excel_path)
