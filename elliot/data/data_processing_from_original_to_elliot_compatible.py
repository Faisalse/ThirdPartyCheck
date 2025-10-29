from pathlib import Path
import pandas as pd


HERE = Path(__file__).resolve()


ROOT = HERE.parents[2]

# used data......
used_data = "amazon-book" # amazon-book / gowalla
# Folder that has the CSVs
if used_data == "gowalla":
    CSV_DIR = ROOT / "neural_graph_collaborative_filtering" / "Data" / used_data

if used_data == "amazon-book":
    CSV_DIR = ROOT / "neural_graph_collaborative_filtering" / "Data" / used_data


print(f"Used data: {used_data}")

# If you know the file name:
train_path = CSV_DIR / "train.txt" 
test_path = CSV_DIR / "test.txt"  



def elliot_format(path):
    users = list()
    item_id = list()
    with open(path, "r") as f:
        for line in f:
            temp = line.strip()
            temp = temp.split()

            item_id.extend(temp[1:])
            users.extend( [temp[0] for i in range(len(temp[1:]))])

    df = pd.DataFrame()
    df["UserID"] = users
    df["ItemID"] = item_id
    df["Rating"] = [1 for i in range(len(item_id))]
    
    df["UserID"] = df["UserID"].astype("int64")
    df["ItemID"] = df["ItemID"].astype("int64")
    df["Rating"] = df["Rating"].astype("int64")

    return df

def data_info(df):
    print("Numer of users:  "+str(len(train_df["UserID"].unique()) ))
    print("Numer of items:  "+str(len(train_df["ItemID"].unique()) ))
    print(df.info())

# training data
train_df = elliot_format(train_path)
print("Information about training data")
data_info(train_df)

# test data
test_df = elliot_format(test_path)
print("Information about test data")
data_info(test_df)

# save training and test data..........
path = "data/"+used_data+"_ngcf"
path = Path(path)
path.mkdir(parents=True, exist_ok=True)

print(path)
train_df.to_csv(path / "train.tsv", sep = "\t", index = False)
test_df.to_csv(path / "test.tsv", sep = "\t", index = False)
print("Completed")



