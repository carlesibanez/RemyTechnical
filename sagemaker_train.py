import sagemaker as sage
import boto3
from sagemaker.estimator import Estimator


client = boto3.client("sts")
account = client.get_caller_identity()["Account"]
region = "eu-south-1"
image_name = "image-segmentation-eggs-and-pans"

ecr_image = "{}.dkr.ecr.{}.amazonaws.com/{}:latest".format(account, region, image_name)

print(f'Using image: {ecr_image}')

# Define data location
data_location = "s3://image-segmentation-eggs-and-pans/train/"

# Define role
# role = sage.get_execution_role()
role = 'AmazonSageMaker-ExecutionRole-20231120T115166'

# Define hyperparmeters
hyperparameters = {"epochs": 200, "batch-size": 4, "num-workers": 4}

instance_type = "ml.g4dn.xlarge"

boto_sess = boto3.session.Session(region_name=region)
sess = sage.Session(boto_session=boto_sess, default_bucket="image-segmentation-eggs-and-pans")

estimator = Estimator(
    role=role,
    instance_count=1,
    instance_type=instance_type,
    image_uri=ecr_image,
    hyperparameters=hyperparameters,
    metric_definitions=[
        {"Name": "train:loss", "Regex": ".*Train loss:\s(.*?)$"},
        {"Name": "val:loss", "Regex": ".*Validation loss:\s(.*?)$"}
    ],
    sagemaker_session=sess,
)

estimator.fit(data_location)