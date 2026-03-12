import pandas as pd

labels_path = r"E:\anemia-ai\dataset\processed\labels.csv"

df = pd.read_csv(labels_path)

def clean_hb(value):

    value = str(value).replace(",", ".").strip()

    try:
        return float(value)
    except:
        return None


df["hb"] = df["hb"].apply(clean_hb)

df = df.dropna(subset=["hb"])

df.to_csv(labels_path, index=False)

print("Labels cleaned successfully")
print("Remaining samples:", len(df))