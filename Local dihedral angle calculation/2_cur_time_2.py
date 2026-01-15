import numpy as np
import pandas as pd

def curdata(content):
    extracted_data = []
    second_line_data = content[1]
    for line in content[num_ni+9:]:
        words_line = line.split()
        if len(words_line) >= 14:
            extracted_data.append(words_line[13]+" "+second_line_data+"\n")
    return extracted_data

def process_content(content):
    result_lines=content
    result_lines=curdata(result_lines)
    file = open(output_filename, 'a', encoding='utf-8')
    file.writelines(result_lines)

def read_content_between_flags():
    content_between_flags = ["ITEM: TIMESTEP"]
    output = []
    flag = "ITEM: TIMESTEP"
    recording = False
    dump = open("1_cur_data_1.txt", "r", encoding='utf-8')
    for line in dump:
        if flag in line:
            if recording:
                process_content(content_between_flags)
                content_between_flags = ["ITEM: TIMESTEP"]
            recording = not recording
        elif recording:
            content_between_flags.append(line.strip())
    if content_between_flags:
        process_content(content_between_flags)

num_ni=38
output_filename = '2_cur_time_2.txt'
read_content_between_flags()