# Remove Ni atoms
input_file = "ni.lammpstrj"
output_file = "1_step.lammpstrj"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    block = []
    remove_count = 0

    for line in infile:
        if line.startswith("ITEM: TIMESTEP"):

            if block:

                new_block = block[:9]
                
                for entry in block[9:]:
                    columns = entry.split()
                    if len(columns) > 1 and columns[1] != '1':
                        new_block.append(entry)
                    else:
                        remove_count += 1
                
                if len(new_block) > 3:
                    count_line = new_block[3].strip()
                    new_count = int(count_line) - remove_count
                    new_block[3] = f"{new_count}\n"
                
                outfile.writelines(new_block)
                
                block = []
                remove_count = 0
            
            block.append(line)
        else:
            block.append(line)
    
    if block:
        new_block = block[:9]
        for entry in block[9:]:
            columns = entry.split()
            if len(columns) > 1 and columns[1] != '1':
                new_block.append(entry)
            else:
                remove_count += 1
        
        if len(new_block) > 3:
            count_line = new_block[3].strip()
            new_count = int(count_line) - remove_count
            new_block[3] = f"{new_count}\n"

        outfile.writelines(new_block)

print(f"Processing complete. The results have been saved to {output_file}")
