import pandas as pd
import numpy as np
import os, os.path

csv_read_path = r"Data\FaceTracker\preprocessed\csv"
csv_write_path = r"Data\FaceTracker\preprocessed\csv\\"

csv_list = os.listdir(csv_read_path)
number_of_csv_files = len(csv_list)

step = 1

if __name__ == "__main__":

    for csv in csv_list:
        if "_fill" in csv:
            break
        else:
            df = pd.read_csv(csv_read_path + "/" + csv)
            # print head for testing
            # print(csv)
            # print(df.head(5))

            idx = np.arange(df["Frame"].min(), df["Frame"].max() + step, step)
            df = df.set_index("Frame").reindex(idx).reset_index()

            # check if it worked
            # print(csv)
            # print(df.head(10))

            df.interpolate(method="linear", inplace=True)
            # print(csv)
            # print(df.head(10))
            # break

            name, type = csv.split(sep=".")
            csv = name + "_fill." + type

            df.to_csv(path_or_buf=csv_write_path + csv, index=False)