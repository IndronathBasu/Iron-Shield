from fastapi import FastAPI,UploadFile
from inference.predict_hb import predict

app = FastAPI()

@app.post("/predict")

async def predict_hb(file:UploadFile):

    path = "temp.jpg"

    with open(path,"wb") as f:

        f.write(await file.read())

    hb = predict(path)

    return {"hemoglobin":hb}