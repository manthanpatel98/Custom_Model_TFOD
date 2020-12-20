# Custom_Model_TFOD
Creating Custom model with TFOD (Faster-RCNN-Inception-v2-coco)

#### For TFOD installation follow github repo [TFOD_Installation&Object_Detection](https://github.com/manthanpatel98/TFOD_Installation-Object_Detection)


## **3 initial steps required:**
* **Creating Dataset with Annotation files** (.xml for TFOD) [**utils**](
* **Downloading Pre-Trained Model weights from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)**
* **Downloading official Github repo of TF-1 OD:** https://github.com/tensorflow/models/tree/v1.13.0



### **Creating Annotated Dataset:**
#### **LabelImg:**
* To annotate our image we can use [LabelImg](https://github.com/tzutalin/labelImg). Follow the steps mentioned to install LabelImge in the official repository.
* LabelImg is a graphical image annotation tool. It is written in Python and uses Qt for its graphical interface. Annotations are saved as XML files in PASCAL VOC format, the format used by [ImageNet](http://www.image-net.org/). Besides, it also supports YOLO format.
* After Annotation, we will have images with annotation files



### **Dataset:**
* Devide the Dataset in 80%-20% train-test or in any other acceptable ratio.
* Form folder structure as **images** ==> **train and test folders**.
* For this example project, you can use [Fruit OD Dataset](https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection) from [Kaggle](https://www.kaggle.com/).
* Here, there are **240 images** in **train folder** and **60 images** in **test folder** which is **not a desirable number for an object detection project** but it is **sufficient for learning how to develop custom models**.
* Dataset has **3 classes:** apple, orange and banana.
* Make sure that **number of images is same as annotated files**.



### **Downloading Pre-Trained Model weights:**
* Any model weights can be downloaded from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). Here, I have used **faster_rcnn_inception_v2_coco model** weights. 

### Downloading official Github repo of TF-1 OD: 
Refer to [TFOD_Installation&Object_Detection](https://github.com/manthanpatel98/TFOD_Installation-Object_Detection) If you are downloading this for first time because I have made several change in the **research folder** then follow below steps.

---

## **Steps to Follow**:
**1>  Copy training, generate_tfrecord.py and xml_to_csv.py from [utils](https://github.com/manthanpatel98/Custom_Model_TFOD/tree/main/utils), Downloaded pre-trained model folder and created images folder to research folder in models.**

**2> Make changes in generate_tfrecord.py and labelmap.pbtxt according to your dataset.** (if you are using same dataset as mentioned then there is no need to change.)

**3> In Anaconda prompt goto research folder and convert .xml to .csv:** This will aggregate all xml files into single csv. 
  
    python xml_to_csv.py
    
* This will create train_labels & test_labels in images folder.

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(333).png" width=650>
    
**4> Convert csv to tfrecords for train and test dataset by executing following commands in anaconda propmt:** This will create train.record & test.record in research folder. These files are faster in execution as compare to csv or xml.

    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
    python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

**5> Copy model's config file [\research\object_detection\samples\configs]** (depends on your model)  **to training & Make 7 Changes in config file:** Config file is the architectural file of the model that we are going to use in pre-training.

*  **num_classes:** No. of classes 

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(334).png" width=650>

*  **fine_tune_checkpoint:** path to pretrained model (downloaded model) folder For that move your downloaded model folder to research

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(335).png" width=650>

*  **num_steps:** Good to use Around 1000/5000 for learning, for project 50000, for production around 200000. **(Here, I have used 1000)**
*  **train_input_reader:** ==> **input_path: & label_map_path:**
*  **eval_input_reader:** ==> **input_path: & label_map_path:** 

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(336).png" width=650>

**6> To Start the training with need train.py, config file and labelmap.pbtxt:**
* Copy/Move **train.py** [research\object_detection\legacy] to **research folder**.
* Copy/Move **deployment** and **nets** folders from [\research\slim] to **research folder**.
* Run Following code to start training:

      python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
      

* Here, I have got good **loss around 0.08**.

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(323).png" width=650>
      
      
* After training **ckpt files** will be generated in **training folder**.

**7> Convert ckpt to PB (inference_graph):**
* Copy/Move **export_inference_graph.py** [research\object_detection] from **object_detection** folder to **research** folder.
* Run Following command for conversion:

**Note:** Make sure to make a change in below code, specify your **model.ckpt-....** (num_steps that you want to consider) before executing.

	  python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1000 --output_directory inference_graph
	  

---


## **Predicting Results:**

* Open **"Object_Detection_tutorial.ipynb"** in jupyter notebook.

* Make changes in Model Preparation as shown below:

<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/Screenshot%20(337).png" width=650>

* Move some images to **object_detection\test_images** for testing and run cells for prediction.

### **Results:**


<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/img1.png" width=400>


<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/img6.png" width=400>


<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/img7.png" width=400>


<img src="https://github.com/manthanpatel98/Custom_Model_TFOD/blob/main/Custom_TFOD_images/img9.png" width=400>


* Model can be trained for **more num_steps** & with **more images** to increase the accuracy.























