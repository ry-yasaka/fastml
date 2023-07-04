from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
from pathlib import Path
import prediction as pd

app = FastAPI()

@app.get('/index')
def hello_world(name: str):
    return f"Hello {name}!"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):# ... 可変長引数(引数の数を指定しない)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = pd.read_image(await file.read())
    image = pd.preprocess(image)

    pred = pd.predict(image)
    print(pred)
    return pred

# @app.get('/results')
# async def results():
#     p = Path('results')
#     # results/yyyymmdd_hhmmss/(png|jpg)
#     result_files = [str(pp) for pp in p.glob('*/*')]
#     return result_files


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
