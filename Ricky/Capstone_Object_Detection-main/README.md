# Welcome
 This repository is the final Capstone Project for Data Science & Artificial Intelligence program

**Notes**: Tensorflow model on custom data (Transfer Learning).The objective is to identify Formula One Racing Team. Please contact me if you are interested in the inference graph and custom dataset. 

**Instructions**: Download and install tensorflow model. The files have been converted to Jupyter Notebook for readability. However, i do recommended to use the python files for implementation.

___

1. Clone the master branch of Tensorflow models repository
>git clone https://github.com/tensorflow/models.git

2. Install protobuf
>conda install -c anaconda protobuf

3. Compile Protobufs
>cd models/research
>
>protoc object_detection/protos/*.proto --python_out=.

4. Install Tensorflow Object Detection Library
>cd object_detection/packages/tf2
>
>python setup.py

5. Test if you have correctly install all the library
>cd object_detection/builders
>
>python model_builder_tf2_test.py

 You are ready to go if you see this code:
> Ran 20 tests in 13.823s
>
>OK (skipped=1)

6. Clone my repository on a separate folder
>git clone https://github.com/RickFSA/Capstone_Object_Detection.git

7. Go to the directory of the folder to open the jupyter notebook 
>jupyter notebook Object_detection_image.ipynb


From this notebook you need to specify the path to the trained model inference_graph/saved_model (370MB)

It should take 15s to load the model, then you are ready to use the model to predict any image from the F1 Formula images. 

I have included some images & video for testing.

Please send a request to ricky.nguyen558@gmail.com for the `dataset` & `inference graph`. 

_____

# Train on  custom dataset
___
1. Generate csv from xml:
>python xml_to_csv.py

2. Adjust class label from generate_tfrecord.py
>code 35 from files 


3. Generate TFRecords from csv:
>python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
>
>python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

4. Config files
> chanage input, label, model_checkpoint


5. Model training:
>python model_main_tf2.py<br> --pipeline_config_path=training/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.config --model_dir=training --alsologtostderr

6. Tensorboard:
>tensorboard --logdir=training/train

7. Extract inference graph (change the config to your selected model):
>python exporter_main_v2.py --pipeline_config_path training/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.config --trained_checkpoint_dir training --output_directory inference_graph
