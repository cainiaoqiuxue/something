import json
import pandas as pd

res = {
    'read': [],
    'comment': [],
    'title': [],
    'author': [],
    'time': [],
}
with open('data.txt','r') as f:
    data=f.readlines()

for d in data:
    a=json.loads(d)
    for key in a:
        res[key].append(a[key])
df=pd.DataFrame(res)
df.drop_duplicates(inplace=True)
df.to_excel('data.xlsx')