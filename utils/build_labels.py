import pandas as pd
import os

india_path = r"E:\anemia-ai\dataset\raw\India\India.xlsx"
italy_path = r"E:\anemia-ai\dataset\raw\Italy\Italy.xlsx"

# Read Excel files
india = pd.read_excel(india_path)
italy = pd.read_excel(italy_path)

print("India rows:", len(india))
print("Italy rows:", len(italy))

# Standardize column names
india.columns = india.columns.str.lower().str.strip()
italy.columns = italy.columns.str.lower().str.strip()

print("India columns:", india.columns)
print("Italy columns:", italy.columns)

# Rename columns
india = india.rename(columns={
    "number": "image",
    "hgb": "hb",
    "gender": "sex",
    "age": "age"
})

italy = italy.rename(columns={
    "number": "image",
    "hgb": "hb",
    "gender": "sex",
    "age": "age"
})

# Add source
india["source"] = "India"
italy["source"] = "Italy"

# Combine datasets
data = pd.concat([india, italy], ignore_index=True)

# Remove rows without Hb
data = data.dropna(subset=["hb"])

# Convert folder number to image filename
data["image"] = data["image"].astype(int).astype(str) + ".jpg"

# Select needed columns
data = data[["image", "hb", "age", "sex", "source"]]

# Save output
output_dir = r"E:\anemia-ai\dataset\processed"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "labels.csv")

data.to_csv(output_file, index=False)

print("Total rows saved:", len(data))
print("labels.csv created at:", output_file)