# Sample Custom Inference with Amazon SageMaker and Hugging Face Transformers


This Repository contains a example for using a custom `inference.py` with Amazon SageMaker and the Hugging Face Inference Toolkit for Amazon SageMaker. 
The custom `inference.py` is located in `code/`. The Endpoint is created using the `deploy.py` script.  
The `deploy.py` also includes an `create_archive` method, which can load a model from hf.co/models, copy your inference `code/` and create a archive `model.tar.gz` for you. You can upload or use this archive with the `deploy.py` script.  
The `deploy.py` script can be used to test your endpoint locally or to deploy it to sagemaker. For testing it locally you need to change the var `LOCAL` in the `deploy` method to `TRUE`. 
