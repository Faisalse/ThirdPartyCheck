import time
from pathlib import Path
from elliot.run import run_experiment
import pandas as pd

file_name = "amazon-book_ngcf"


start = time.time()
run_experiment("config_files/"+file_name+".yml")
end = time.time()

path = "results/"+file_name
path = Path(path)
path.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame()

df["Training_time"] = [str(end - start)+"s"]
df.to_csv(path / "training_time.csv", sep="\t", index = False)


