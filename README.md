## Detecting and analyzing incorrect model predictions with Amazon SageMaker Model Monitor and Debugger

This repository contains the notebook and scripts for the blogpost "Detecting and analyzing incorrect model predictions with Amazon SageMaker Model Monitor and Debugger" 

[Create a SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html) and clone the repository:
```
git clone git@github.com:aws-samples/amazon-sagemaker-analyze-model-predictions.git
```

In the notebook [analyze_model_predictions.ipynb](analyze_model_predictions.ipynb) we first deploy a [ResNet18](https://arxiv.org/abs/1512.03385) model that has been trained to distinguish between 43 categories of traffic signs using the [German Traffic Sign dataset](https://ieeexplore.ieee.org/document/6033395).

We will setup [SageMaker Model Monitor](https://aws.amazon.com/blogs/aws/amazon-sagemaker-model-monitor-fully-managed-automatic-monitoring-for-your-machine-learning-models/) to automatically capture inference requests and predictions.
Afterwards we launch a [monitoring schedule](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-scheduling.html) that periodically kicks off a custom processing job to inspect collected data and to detect unexpected model behavior. 

We will then create adversarial images which lead to the model making incorrect predictions. Once Model Monitor detects this issue, we will use [SageMaker Debugger](https://aws.amazon.com/blogs/aws/amazon-sagemaker-debugger-debug-your-machine-learning-models/) to obtain visual explanations of the deployed model. This is done by updating the endpoint to emit tensors during inference and we will then use those tensors to compute saliency maps.  

The saliency map can be rendered as heat-map and reveals the parts of an image that were critical in the prediction. Below is an example (taken from the  [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)):  the image on the left is the input into the fine-tuned ResNet model, which predicted the image class 25 (‘Road work’). The right image shows the input image overlaid with a heat-map where red indicates the most relevant and blue the least relevant pixels for predicting the class 25.

<p>
<img src="images/image.png", width="500" height="250" />
</p>

## License

This project is licensed under the Apache-2.0 License.

