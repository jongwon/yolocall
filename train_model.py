from pydantic import BaseModel

class TrainingInfo(BaseModel):
    trainingId: str =''
    name: str = ''
    epochs = 300
    imageSize = 640
    batchSize = 8
    weights: str = "yolov5m.pt"
    hyp: str = "data/hyps/hyp.scratch.yaml"
    data: str = ''
    deviceForTorchscript: str = "cpu"

