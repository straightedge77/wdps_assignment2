
import json

with open("./web.json", 'r') as rf:
    lines = rf.readline()

s = json.loads(lines)

for item in s["data"]:
    item["label"]["show"] = "true"


with open("./refined_web.json", 'w') as wf:
    json.dump(s, wf)
