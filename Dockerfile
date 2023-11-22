ARG REGION=eu-south-1

# SageMaker PyTorch image
FROM 692866216735.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu121-ubuntu20.04-ec2

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /src /opt/ml/code

# Install dependencies
RUN pip install -r /opt/ml/code/requirements.txt
RUN pip install sagemaker-training

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM train.py