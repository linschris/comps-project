import csv

def convert_csv_to_dict(file_path: str) -> dict:
    values = {}
    with open(file_path) as csv_file:
        csv_file = csv.reader(csv_file)
        for i, row in enumerate(csv_file):
            if i > 0:
                values[int(row[0])] = row[3]
    
    return values

                



print(convert_csv_to_dict("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/vocab.csv")[3842])