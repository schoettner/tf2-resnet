# Tensorflow2 Classification CNN Training Template #
Due to major changes in the API from TF1 to TF2, I want to provide a template for training with generic models. The example
will be done with RESNET50.
Earlier versions of this repo also included alternative training methods. However, since TF continues to shift everything
towards Keras, there is no need to keep them updated anymore

```bash
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=tf2_training_$DATE
export GCS_JOB_DIR=gs://ri-training-log-temp/jobs/$JOB_NAME  # Change your BUCKET
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