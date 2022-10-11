import json

def write_json(items,fp):
    with open(fp, 'w') as file:
        json.dump(items, file)
        
def read_json(fp):
    with open(fp,'r') as file:
        data = json.load(file)
    return data
