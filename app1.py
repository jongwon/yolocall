from fastapi import FastAPI
from pydantic import BaseModel
from image_process import train_start
import os
from train_model import TrainingInfo

app = FastAPI()

class Resp(BaseModel):
    result: bool = True
    message: str = ''

@app.post("/trainModel")
async def trainModels(train: TrainingInfo):
    train_start(train)
    resp=Resp()
    resp.result = True
    resp.message= 'success'
    return resp

