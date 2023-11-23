# RemyTechnical
This repository contains my implementation for the Remy Robotics Technical Assignment.

The assignment consisted on traning a computer vision model to perform semantic segmentation.
In order to train the model a series of images and per-pixel masks were given. The masks indicate two types of objects: eggs and frying pans. The different classes are encoded as the following pixel values:
* 0: background
* 128: egg
* 255: frying pan 

## Code structure

The code in the repository is structured as follows:
* [```src```](/src) directory: contains the source code of the solution developed here
    * [```models```](/src/models/) directory: contains the file diffining the U-net model.
    * [```utils```](/src/utils/) directory: contains the definition of the Dataset object used to load and preprocess the data.
    * [```train.py```](/src/train.py) file: this file contains the initialization of the model and the training.
    * [```test.py```](/src/test.py) file: this file contains the code to load a model and run the inference on the test set.
* [```Dockerfile```](Dockerfile): this file contains the definition to create a docker image of the repo.
* [```build_and_push.s```](build_and_push.sh): this file invokes the creation of the docker image and pushes it to an [Amazon Elastic Container Registry (ECR)](https://aws.amazon.com/es/ecr/). 
* [```sagemaker_train.py```](sagemaker_train.py): this file is used to launch a AWS Sagemaker job using the previously built docker image. (This requires first uploading the training data to an AWS S3 bucket).

## To run the code

In order to run the code, the ```sagemaker_train.py``` can be used, or else directly invoquing ```python src/train.py``` with the correct parameters.

## The solution

The solution used to tackle the problem of semantic image segmentation is based in the U-Net architecture presented [here](https://arxiv.org/abs/1505.04597).

The U-Net architecture is a well known architecture, initially presented for medical image segmentation. It receives its name from the shape of the model. First, it uses Convolutional Neural Networks (CNNs) to generate feature maps to represent high level features of the images. Then it uses these high level features to reconstruct the segmentation mask. Additionally, it uses what is refered as _skip connections_, which are results from the compression phase of the model, which have proven to help preserve spatial resolution, by keeping the hight frequencies.

To optimize the model, the cross entropy between the predicted mask and the ground truth was used.

### Other considerations: data preprocessing

After initial inspection, it was seen that the given images were not all the same size, and not the same format (same for the masks). Therefore a preprocessing step was added to make the data suitable for training the model.

Regarding the different sizes it was decided to pad all the images in order to make them squared. Then all the images can get resized to 400x400 pixels (user set argument in the _Dataset_ object) to ease the computations (instead of using the full resolution version). For the testing, masks are cropped and upsampled to the original images resolution.

The different encoding formats, however, affected on the preprocessing of the masks. As it is expected, the masks in .png kept the different values (0, 128 and 255) intact. On the other hand, when using compression, such as in jpeg, values get slightly distorted, which was corrected by clipping the values to 0, 128 and 255.


## Unsupervised learning solution.

In case the masks were not provided, the problem would need to be solved using different methods. Some initial ideas that would be worth exploring are:
* K-means clustering. This could be performed using pixel level features to cluster similar pixels together (e.g. similar color, intensity...). This could also be applied to high level features that could first be extracted from the images. Then this lattent representation could be used for the clustering task.
* Use pretrained segmentation models. For example, the [Segment Anything Model](https://arxiv.org/abs/2304.02643) could be used to segment the images and obtain the different objects present. Then som clustering technique could be used to group together different instances of pans and eggs. 
* Semisupervised approach. Initially some images would need to be manually labeled. Then the model here presented could be used, coupled with a GAN to generate the masks. Then the newly generated masks could be retrofitted in to the U-net model to learn more features. 
