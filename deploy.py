import os
import boto3
from distutils.dir_util import copy_tree
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.local import LocalSession
from sagemaker import Session


from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def deploy():
    INITIAL_INSTANCE_COUNT = 1
    INSTANCE_TYPE = "ml.g4dn.xlarge"
    MODEL_DATA = "s3://hf-sagemaker-inference/example_custom_inference/model.tar.gz"  # TODO: change to your s3 path
    REGION = "us-east-1"
    LOCAL = False  # TODO: change this to False if you want to test locally first
    os.environ["AWS_DEFAULT_REGION"] = REGION
    if LOCAL:
        sess = LocalSession()
        sess.config = {"local": {"local_code": True}}
        ROLE_NAME = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"
        INSTANCE_TYPE = "local"
        MODEL_DATA = f"file://{os.getcwd()}/model/model.tar.gz"
    else:
        iam_client = boto3.client("iam")
        ROLE_NAME = iam_client.get_role(RoleName="sagemaker_execution_role")["Role"][
            "Arn"
        ]  # TODO: change to your role_name
        sess = Session()

    hf_model = HuggingFaceModel(
        model_data=MODEL_DATA,
        role=ROLE_NAME,
        transformers_version="4.6.1",
        pytorch_version="1.7.1",
        py_version="py36",
    )

    predictor = hf_model.deploy(initial_instance_count=INITIAL_INSTANCE_COUNT, instance_type=INSTANCE_TYPE)

    # For automation
    # result = predictor.predict({"inputs": "I love the new Amazon SageMaker Hugging Face Container"})
    # print(result)

    # predictor.delete_endpoint()


def create_archive(model_name):
    tmp_dir = os.path.join(os.getcwd(), "model")
    AutoTokenizer.from_pretrained(model_name).save_pretrained(tmp_dir)
    AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(tmp_dir)
    copy_tree(os.path.join(os.getcwd(), "code"), (os.path.join(tmp_dir, "code")))
    os.popen(f"cd {tmp_dir} && tar zcvf model.tar.gz *")


if __name__ == "__main__":
    # TODO: comment in if you want to create an archive
    # uploading is not included.
    # create_archive(MODEL_NAME)
    deploy()
