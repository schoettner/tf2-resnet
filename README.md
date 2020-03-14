# TensorFlow 2.x Classification CNN Training Template #
Due to major changes in the API from TF1 to TF2, I want to provide a template for training models with the current TensorFlow version (2.1 on last update). 
The example will be done with RESNET50 as base-model, but you can replace it with whatever model you like.
Earlier versions of this repo also included alternative training methods. However, since TF continues to shift everything
towards Keras, there is no need to keep them updated anymore.
Instead this code displays how to setup a training that runs both on your local machine, cloud VMs or serverless on Google's *AI Platform*.

### Run locally ###
To test your code, first start with the --test flag to shorten your training. The download of the dataset is only done once.
So the second run will be a lot faster. The setup for a cloud VM would be the same.
```bash
git checkout git@github.com:schoettner/tf2-resnet.git
cd tf2-resnet
pip3 install -r requirements.txt
python3 run.py --job-dir /tmp --test
```

### Run on AI Platform ###
If you do not want to train on a local machine or cloud VM, Ai Platform is the tool to use. It is a GCP product that
allows serverless training. It also comes with several benefits like hyperparameter tuning (instead of Keras-Tuner). I strongly suggest using this
tool over local training if costs are not an issue. 
```bash
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=tf2_training_$DATE
export GCS_JOB_DIR=gs://<your-bucket>/jobs/$JOB_NAME  # Change your BUCKET
export EPOCHS=2
export REGION=europe-west4
gcloud ai-platform jobs submit training $JOB_NAME \
    --region $REGION \
    --stream-logs \
    --config config.yaml \
    --package-path trainer \
    --module-name trainer.task \
    --job-dir $GCS_JOB_DIR \
    -- \
    --epochs $EPOCHS \
    --test
```
When your job is finished, your model/weights are exported and you can check your training log with the cloud shell by typing
```bash
tensorboard --logdir gs://<your bucket>/jobs/tf2_training<timestamp>/<machine id>/training/tensorboard
```
and change the preview port to *6006* to check your TensorBoard.

### Customization ###
As you can see in the *trainer/task.py/parse_arguments()* there are already plenty of parameters you can adjust to your 
training to fit your needs. However, for an actual project you still need to do several changes that are beyond this demo.
The most obvious being the change of the dataset and transfer learning. For image datasets, TensorFlow Records are
the way to go. I will reference sources where you can find information about the creation of such datasets (e.g. with
TensorFlow Extended)

### Not included ###
Here I want to list some information that is intentionally not included in this project that you want to take a look at, when
you want to start your own project
- tf.data: API to build input pipelines for your training
- TensorFlow Transform: Data preprocessing and end-to-end training
- Hyperparameter tuning: Parallel computing to optimize your training's parameters
- TensorFlow Profiler: Observe your training to find bottlenecks and get suggestions how to fix them
- Transfer Learning: Use pre-trained models instead of training from scratch
- Model versioning: Save and deploy models in production. Also required for transfer learning

### References ###
[TensorFlow Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)  
[AI Platform](https://cloud.google.com/ai-platform/docs)  
[Hyperparameter tuning](https://cloud.google.com/ai-platform/training/docs/hyperparameter-tuning-overview)
[TF Transform](https://www.tensorflow.org/tfx/transform/get_started)   
[tf.data](https://www.tensorflow.org/guide/data)  
[Tensorflow Hub](https://www.tensorflow.org/hub)  

