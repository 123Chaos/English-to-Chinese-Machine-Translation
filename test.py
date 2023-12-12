import json


def get_json_list(file_path):
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

train_src = get_json_list('./train/src.jsonl')

