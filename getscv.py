import os
import csv

base_path = './agent/test_analysis'
output = []

for i in range(1, 51):
    folder_name = f'P{i:03d}'
    file_path = os.path.join(base_path, folder_name, f'{folder_name}_analysis_summary.txt')
    
    print(f'Processing {file_path}')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # print(line)
                if line.startswith('Publishable:'):
                    publishable_status = line.split(':')[1].strip()
                    output.append({"id": folder_name, "publishable": publishable_status})
                    break

print(output)
with open('output.csv', 'w', newline='') as csv_file:
    fieldnames = ['id', 'publishable']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(output)