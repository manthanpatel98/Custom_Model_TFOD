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
* Make sure that **number of images is same as annotated files**.



### **Downloading Pre-Trained Model weights:**
* Any model weights can be downloaded from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md). Here, I have used **faster_rcnn_inception_v2_coco model** weights. 

### Downloading official Github repo of TF-1 OD: 
If you 

---

## **Steps to Follow**:
**1>  Copy training, generate_tfrecord.py and xml_to_csv.py from [utils](https://github.com/manthanpatel98/Custom_Model_TFOD/tree/main/utils) and created images folder to research folder in models.**

**2> Make changes in generate_tfrecord.py and labelmap.pbtxt according to your dataset.** (if you are using same dataset as mentioned then there is no need to change anything.)

**3> In Anaconda prompt goto research folder and convert .xml to .csv:** This will aggregate all xml files into single csv. This will create train_labels & test_labels in images folder.
  
    python xml_to_csv.py
    
**4> Convert csv to tfrecords for train and test dataset by executing following commands in anaconda propmt:**

    python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
	  python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record





















