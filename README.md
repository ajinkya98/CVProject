# Computer Vision [CSCI B657] Project
## Segmentation and Classification for Organ Cell Microscopic Images
### Project Description:
Deep Learning and Computer Vision is rising in popularity in the Biomedical Image Analysis Domain. In the past few years, a plethora of literature has been published in this field leveraging the power of AI and machine learning to better understand the human body and brain. From cancer detection to tumor segmentation, machine learning has vastly changed the landscape of biomedical imaging. Through this project we propose to apply techniques learned in the computer vision class to perform segmentation and classification on organ cell images taken by electron microscopes. The idea of the project is to be able to help medical professionals better diagnose the ailments of the human body. In a traditional setting a lab expert would be sitting at his desk individually analyzing the sliced images of the cell to classify or segment them. Automating the process would help not only speed up the process of recognizing the organ to which a particular cell belongs to but also give a high accuracy output.

Some of the plausible application we see for it are in the blood culture labs wherein blood cells found in urine samples can help identity the organ that they have come from, knowing which the doctor with domain expertise in that particular organ can diagnose and help treat the particular patient. Some of the prominent challenges of a project when it comes to the biomedical domain is that it leaves very little room for error since the results will directly impact the patients. When a classification task is involved in the biomedical domain it is important to give more weightage to the positive class. For example, if someone is detected with breast cancer, we would want the false positive rate to be a little higher because we do not want to miss any patients with cancer as chemotherapy has a strict deadline to start. As for the misclassified patients more tests can further be done to be surer about the cancer classification. Thus, in this project we will brainstorm ideas to tackle this problem in our cell classification and segmentation task.

Our end goal is to have a state-of-the-art model for both classification and segmentation of organ cells to accurately determine the organ based on the electron microscope sliced image. For the image segmentation task, we plan to use the Unet architecture as it is popularly used in biomedical image segmentation and is reviewed to give impeccable performance on biomedical datasets. For the classification task we plan on using the popular ResNet architecture and its variants to draw comparative analysis among the models. For the frameworks to be used, PyTorch will be used for the classification task and TensorFlow will be the choice of framework for the segmentation task as different group members are comfortable using different deep learning frameworks.
### Dataset Details:
The dataset has been acquired from the CNS lab at IU. The total size of the dataset if 45.1 GB. The dataset used for the classification and segmentation will be the same except that the annotations file will not be used for the classification. The dataset comprises of microscope cell images taken from the following organs:

1. colon
2. endotherium_1
3. endotherium_3
4. kidney
5. liver
6. lung
7. lymph node
8. pancreas
9. skin_1
10. skin_2
11. small intestine
12. spleen

There are 1200 training images and 600 test images. The annotations file has been provided as well for the segmentation task.

The segmentation dataset consist of 5 Images for training and 2 images for testing. In order to increase dataset size, we have divided the images into patches for training.

### Hyperparameter Sweep[Clasification]:

![image](https://user-images.githubusercontent.com/55157425/166335580-e25e86b4-03c7-4d43-a4fe-93ec7a5e1448.png)

### Results With an Interactive DashBoard[Classification And Segmentation]:

![WhatsApp Image 2022-05-02 at 6 51 37 PM](https://user-images.githubusercontent.com/32778343/167050909-355924e1-6a2f-4a8c-ba9f-5f3938fa1101.jpeg)

<img width="1070" alt="Screen Shot 2022-05-05 at 6 34 51 PM" src="https://user-images.githubusercontent.com/32778343/167050938-694b73b2-e6bf-46fd-b6db-c61d8edd85a4.png">

https://wandb.ai/ajinkya98/pytorch-cell-classification/reports/Classification-for-Organ-Cell-Microscopic-Images--VmlldzoxNzgzODEy

### Tools Used:
#### Classification:

Data Prep and Analysis: <b>Numpy</b>, <b>Pandas</b>, <b>Matplotlib</b>, <b>Seaborn</b>

Deep Learning Model and Experimentation: <b>Pytorch</b>

DashBoard, Runs Management and Visualizations: <b>Weights and Biases</b>


### References:

Idea of how to approach the segmentation problem was taken from this youtube channel- 

https://github.com/bnsreenu/python_for_microscopists 

https://www.youtube.com/results?search_query=digital+sreeni
