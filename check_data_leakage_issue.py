import pandas as pd

from pathlib import Path
dataset_name = "Amazon-book"
path = Path("data/"+dataset_name)

train = {}
with open(path/"train.txt") as f:
     for l in f.readlines():
        l = l.strip('\n').split(' ')
        items = l[1:]          
        user_id = l[0]
        train[user_id] = set(items)


test = {}
with open(path/"test.txt") as f:
     for l in f.readlines():
        l = l.strip('\n').split(' ')
        items = l[1:]        
        user_id = l[0]
        test[user_id] = set(items)

user_with_data_leakage = []
for key in train.keys():
    if key in test:
        train_ = train[key]
        test_ = test[key]
        train_.intersection(test_)
        if len(train_.intersection(test_)) > 0:
            user_with_data_leakage.append(key)

print(f"Number of users with data leakage:  {len(user_with_data_leakage)}")
#### No data leakage issue has beeb not found...................................
print(f"train users: {len(train)} test users: {len(test)}")