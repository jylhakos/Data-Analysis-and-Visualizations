# Amazon SageMaker

The process for creating a machine learning (ML) pipeline consists of the following steps.

Step 1. You make raw data available in Amazon Simple Storage Service (Amazon S3), perform exploratory data analysis (EDA), develop the initial ML model and evaluate its inference performance.

Step 2. You integrate the model with runtime Python scripts so that it can be managed and provisioned by Amazon SageMaker AI. You define the logic for preprocessing, evaluation, training, and inference.

Step 3. Define the ML pipeline for the input and output placeholders for each step of the pipeline.

Step 4. Create the pipeline for the underlying infrastructure by using AWS CloudFormation.

Step 5. Run the ML pipeline defined in step 4. You also prepare data or data locations to fill in concrete values for the input/output placeholders that you defined in step 3. This includes the runtime scripts defined in step 2 as well as model hyperparameters.

Step 6. You implement continuous integration and continuous deployment (CI/CD) processes and similar extensions of the ML pipeline.

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/Amazon%20SageMaker/machine_learning_pipeline.png?raw=true)

*Figure: Steps for creating the training and inference pipeline*

To use Amazon SageMaker for Large Language Model (LLM) inference, you'll deploy your LLM to a SageMaker endpoint using a container, like the Large Model Inference (LMI) container. 

LMI container optimizes performance for LLMs, leveraging libraries like vLLM.

Quantizing your model (e.g., using AWQ, GPTQ, or FP8) can reduce memory usage and improve inference speed, but may affect output quality.

**References**

- **What is Amazon SageMaker AI?**: https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html
- **What is Amazon SageMaker Pipelines?**:https://aws.amazon.com/sagemaker-ai/pipelines/
- **Getting started with Amazon SageMaker AI**:https://aws.amazon.com/sagemaker-ai/getting-started/
- **Run Agent**: https://docs.aws.amazon.com/sagemaker/latest/dg/edge-getting-started-step5.html
- **Creating production-ready ML pipelines on AWS**:https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-production-ready-pipelines/welcome.html

