import numpy as np
from scipy.special import perm, comb

#Determine the positional relationship between the carbon atom and the dihedral angle.
def is_point_inside_dihedral(A, B, C, D, E):
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    AD = np.array(D) - np.array(A)
    AE = np.array(E) - np.array(A)
    n1 = np.cross(AB, AC)
    n2 = np.cross(AB, AD)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    sign1 = np.dot(AE, n1)
    sign2 = np.dot(AE, n2)

    if sign1 * sign2 > 0:
        return 0
    else:
        return 1

#Find the line number by ID
def find_row(id,content):
    i=0
    for line in content:
        words = line.split()
        if words[0] == id:
            return i
        i = i + 1

#Calculate the dihedral angle between planes ABC and ABD
def vector_cross_product(v1, v2):
    return np.cross(v1, v2)
def vector_dot_product(v1, v2):
    return np.dot(v1, v2)
def vector_magnitude(v):
    return np.linalg.norm(v)
def calculate_dihedral_angle(A, B, C, D):
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    AD = np.array(D) - np.array(A)
    N1 = vector_cross_product(AB, AC)
    N2 = vector_cross_product(AB, AD)
    N1_normalized = N1 / vector_magnitude(N1)
    N2_normalized = N2 / vector_magnitude(N2)
    cos_theta = vector_dot_product(N1_normalized, N2_normalized)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    angle_degrees = np.degrees(theta)
    return angle_degrees

#local curvature
def curvature(content):
    matrix_ni_position = np.zeros((num_ni, 5))
    second_matrix_ni_position = np.zeros((num_ni, 5))
    content[8] = content[8] + " cur"

    i = num_ni+9

    while i<=int(content[3])+8:
        words_i = content[i].split()
        E = [float(words_i[6]), float(words_i[7]), float(words_i[8])]
        k=0
        j = 9
        while j <= num_ni+8:
            words_j = content[j].split()
            
#Find the two directly bonded Ni atoms
            distance_c_ni = pow(float(words_i[6]) - float(words_j[6]), 2) + pow(float(words_i[7]) - float(words_j[7]),2) + pow(float(words_i[8]) - float(words_j[8]), 2)
            if distance_c_ni <=cutoff_c_ni:
                matrix_ni_position[k][0] = int(words_j[0])
                matrix_ni_position[k][1] = float(words_j[6])
                matrix_ni_position[k][2] = float(words_j[7])
                matrix_ni_position[k][3] = float(words_j[8])
                matrix_ni_position[k][4] = distance_c_ni
                k=k+1
            j=j+1
        filtered_matrix = matrix_ni_position[matrix_ni_position[:, 4] != 0.0]
        sorted_matrix = sorted(filtered_matrix, key=lambda row: row[4])
        matrix_ni_position = np.zeros((num_ni, 5))
        if k >= 2:
            l=0
            m=0
            a = int(sorted_matrix[0][0])
            b = int(sorted_matrix[1][0])
            for line in content:
                words_line = line.split()
                if l >= 9 and int(words_line[0]) != a and int(words_line[0]) != b and l <= num_ni+8:
                    deerta_ni_x_1 = abs(float(words_line[6]) - sorted_matrix[0][1])
                    deerta_ni_y_1 = abs(float(words_line[7]) - sorted_matrix[0][2])
                    deerta_ni_z_1 = abs(float(words_line[8]) - sorted_matrix[0][3])
                    deerta_ni_x_2 = abs(float(words_line[6]) - sorted_matrix[1][1])
                    deerta_ni_y_2 = abs(float(words_line[7]) - sorted_matrix[1][2])
                    deerta_ni_z_2 = abs(float(words_line[8]) - sorted_matrix[1][3])
                    distance_ni_ni_1 = pow(deerta_ni_x_1, 2) + pow(deerta_ni_y_1, 2) + pow(deerta_ni_z_1, 2)
                    distance_ni_ni_2 = pow(deerta_ni_x_2, 2) + pow(deerta_ni_y_2, 2) + pow(deerta_ni_z_2, 2)
#Find the two second-nearest Ni atoms.
                    if distance_ni_ni_1<=cutoff_ni_ni and distance_ni_ni_2<=cutoff_ni_ni:
                        add_distance_ni_ni=distance_ni_ni_1 + distance_ni_ni_2
                        second_matrix_ni_position[m][0] = float(words_line[0])
                        second_matrix_ni_position[m][1] = float(words_line[6])
                        second_matrix_ni_position[m][2] = float(words_line[7])
                        second_matrix_ni_position[m][3] = float(words_line[8])
                        second_matrix_ni_position[m][4] = add_distance_ni_ni
                        m = m + 1
                l=l+1
            second_filtered_matrix = second_matrix_ni_position[second_matrix_ni_position[:, 4] != 0.0]
            second_sorted_matrix = sorted(second_filtered_matrix, key=lambda row: row[4])
            second_matrix_ni_position = np.zeros((num_ni, 5))

        if k>=2 and m == 2:
            A = [sorted_matrix[0][1], sorted_matrix[0][2], sorted_matrix[0][3]]
            B = [sorted_matrix[1][1], sorted_matrix[1][2], sorted_matrix[1][3]]
            C = [second_sorted_matrix[0][1], second_sorted_matrix[0][2], second_sorted_matrix[0][3]]
            D = [second_sorted_matrix[1][1], second_sorted_matrix[1][2], second_sorted_matrix[1][3]]
            angle = calculate_dihedral_angle(A, B, C, D)
            content[i] = content[i] + " " + str(angle)

        if k>=2 and m > 2:
            A = [sorted_matrix[0][1], sorted_matrix[0][2], sorted_matrix[0][3]]
            B = [sorted_matrix[1][1], sorted_matrix[1][2], sorted_matrix[1][3]]
            o=0
            r=0
            second_comb_numb=int(comb(m, 2))
            within_the_dihedral = np.zeros((second_comb_numb, 1))
            matrix_second_comb_numb = np.zeros((second_comb_numb, 1))
            while o<=m-2:
                C = [second_sorted_matrix[o][1], second_sorted_matrix[o][2], second_sorted_matrix[o][3]]
                p = o+1
                while p<=m-1:
                    D = [second_sorted_matrix[p][1], second_sorted_matrix[p][2], second_sorted_matrix[p][3]]
                    within_the_dihedral[r][0] = is_point_inside_dihedral(A, B, C, D, E)
                    angle = calculate_dihedral_angle(A, B, C, D)
                    matrix_second_comb_numb[r][0]=angle                    
                    p=p+1                   
                    r=r+1
                o=o+1
            matrix_second_sort_comb_numb = sorted(matrix_second_comb_numb, key=lambda row: row[0])
            if 1 in within_the_dihedral:

                combined_matrix = np.hstack((within_the_dihedral, matrix_second_comb_numb))
                filtered_rows = combined_matrix[combined_matrix[:, 0] == 1]
                min_value = np.min(filtered_rows[:, 1])
                content[i] = content[i] + " " + str(min_value)
            else:
                content[i] = content[i] + " " + str(matrix_second_sort_comb_numb[second_comb_numb-1][0])

        i=i+1
    n=0
    while n<=int(content[3])+8:
        content[n] = content[n] + "\n"
        n=n+1
    return content

#A function for processing atomic coordination numbers.
def distance(content):
    print(f"时间步：{content[1]}")
    result_lines=[]
    i = 9
    k = 0
    car_net_num = 0
    car_net_num_data=[]
    content[8] = content[8] + " coord_c-c"
    while i<=int(content[3])+8:
        count_coord = 0
        j = 9
        while j <= int(content[3])+8:
            words_i = content[i].split()
            words_j = content[j].split()
            if len(words_i) >= 5 and len(words_j) >= 5 and words_i[4] == '2' and words_j[4] == '2':
                distance_c_c = pow(float(words_i[6])-float(words_j[6]),2)+pow(float(words_i[7])-float(words_j[7]),2)+pow(float(words_i[8])-float(words_j[8]),2)
                if distance_c_c <= cutoff_c_c:
                    count_coord=count_coord+1
            j = j + 1
        content[i] = content[i] + " " + str(count_coord-1)
        i=i+1
    for line in content:
        words = line.split()
        if len(words)>8 and (words[-1] == '3' or words[-1] =="2"or words[-1] =="1"):
            car_net_num = car_net_num+1

        
        if len(words)>8 and (words[-1] == '3' or words[-1] =="0"):
            k = k + 1
            continue
        result_lines.append(line)
    car_net_num_data.append(content[1] + " " + str(car_net_num) + "\n")
    result_lines[3] = str(int(result_lines[3]) - k)
    return result_lines, car_net_num_data

#File output function
def process_content(content):
    result_lines=content
    result_lines, car_net=distance(result_lines)
    result_lines=curvature(result_lines)
    file = open(output_filename, 'a', encoding='utf-8')
    file.writelines(result_lines)
    file = open(car_net_num_filename, 'a', encoding='utf-8')
    file.writelines(car_net)

#Sort the atoms.
def sort_key(line):
        return int(line.split(' ')[0])
def mr_sort(content):
    content_to_sort = content[9:]
    sorted_content = sorted(content_to_sort, key=sort_key)
    sorted_lines = content[:9] + sorted_content
    return sorted_lines

def read_content_between_flags():
    content_between_flags = ["ITEM: TIMESTEP"]
    output = []
    flag = "ITEM: TIMESTEP"
    recording = False
    #work_path = "E:\\ni\\20240622\\chuli\\"
    dump = open("ni.lammpstrj", "r", encoding='utf-8')
    for line in dump:
        if flag in line:
            if recording:
                content_between_flags=mr_sort(content_between_flags)
                process_content(content_between_flags)
                content_between_flags = ["ITEM: TIMESTEP"]
            recording = not recording
        elif recording:
            content_between_flags.append(line.strip())
    if content_between_flags:
        content_between_flags=mr_sort(content_between_flags)
        process_content(content_between_flags)

#Main function
num_ni=38
cutoff_c_c=pow(1.8,2)
cutoff_c_ni=pow(3.0,2)
cutoff_ni_ni=pow(3.5,2)
output_filename = '1_cur_data_1.txt'
car_net_num_filename = '2_car_net_num.txt'
read_content_between_flags()