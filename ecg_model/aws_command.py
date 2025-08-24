import sagemaker
from sagemaker.pytorch import PyTorchModel

role = "<your-sagemaker-execution-role>"  # check IAM

model = PyTorchModel(
    model_data="s3://hackathon-demo-bucket-sc/model/model.tar.gz",
    role=role,
    entry_point="infer.py",
    framework_version="2.0",   # depends on your PyTorch version
    py_version="py310",        # or py39
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)