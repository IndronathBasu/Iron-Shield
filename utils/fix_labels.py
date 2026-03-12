import pandas as pd

labels_path = r"E:\anemia-ai\dataset\processed\labels.csv"

df = pd.read_csv(labels_path)

def build_filename(row):
    number = row["image"].replace(".jpg","")
    source = row["source"]
    return f"{source}_{number}.jpg"

df["image"] = df.apply(build_filename, axis=1)

df.to_csv(labels_path, index=False)

print("labels.csv updated successfully")