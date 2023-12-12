# iwslt2017-en-zh

## Data

- train (10000)
  - src.jsonl
  - target.jsonl
- validation (800)
  - src.jsonl
  - target.jsonl
- test (1000)
  - src.jsonl
  - target.jsonl

## Example

train, src.jsonl, line 4: 

{"text": "A lot of persuasion, a lot of wonderful collaboration with other people, and bit by bit, it worked."}

train, target.jsonl, line 4: 

{"text": "通过大量劝说 与他人通力合作 逐渐 它走上正轨"}

## Load File

```
import json

def get_json_list(file_path):
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list
```
