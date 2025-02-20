from __future__ import annotations

import csv
import os
from glob import glob

import pandas as pd

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(main_dir, "results/guide/")


def main():
    data = []
    for file in glob(RESULTS_PATH + "*.csv"):
        success_count = 0
        with open(file) as f:
            reader = csv.reader(f)
            _header = next(reader)
            for row in reader:
                if row[-1] == "True":
                    success_count += 1

        file_params = f"{os.path.basename(file[:-4])}".split("_")
        if len(file_params) >= 4:
            task, guide, examples, model = file_params[:4]
            data.append([task, guide, examples, model, success_count])

    df = pd.DataFrame(data, columns=["Task", "Guide", "Examples", "Model", "Success"])
    df = df.sort_values(by=["Task", "Examples", "Guide", "Model"])
    # print(df[df['Model'] == 'llama1b'].to_string(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
