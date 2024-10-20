import numpy as np
import onnx
from dotenv import load_dotenv

load_dotenv()

import mlflow

with mlflow.start_run(run_id="3c8c39e244804d8a8251db07b5b6525c"):
    onnx_filename = "epoch=0-val_f1_score=0.46041664481163025.onnx"
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    mlflow.onnx.log_model(onnx_model,
                          "onnx",
                          input_example=np.zeros((1, 3, 224, 224)),
                          conda_env=mlflow.onnx.get_default_conda_env())

