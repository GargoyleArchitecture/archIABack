import json
import glob
import os

files = glob.glob('c:/Users/arnul/proyectos/trabajo/Tesis/archIABack/back/eval/datasets/*.json')
for f in files:
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            data = json.loads(content)
            count = len(data.get('qa_pairs', data) if isinstance(data, dict) else data)
            print(f"{os.path.basename(f)}: {count}")
    except Exception as e:
        print(f"Error reading {os.path.basename(f)}: {e}")
