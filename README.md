# MNIST Handwritten Digits Recognition using Tensorflow


## MNIST dataset

The `MNIST` database is available at http://yann.lecun.com/exdb/mnist/

The `MNIST` database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

![](https://github.com/shrigulhane100/Handwritten-digits-classification-Tensorflow/blob/main/images/samples.png)

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.


## Results
***`A validation dataset of size 12,000 was deduced from the Training dataset with its size being changed to 48,000. We train the following models for 20 epochs.`***

### Prarameters Initialization
* Both models have been initialized with random weights sampled from a normal distribution and bias with 0.
* These parameters have been intialized only for the Linear layers present in both of the models.
* If `n` represents number of nodes in a Linear Layer, then weights are given as a sample of normal distribution in the range `(0,y)`. Here `y` represents standard deviation calculated as `y=1.0/sqrt(n)`
* Normal distribution is chosen since the probability of choosing a set of weights closer to zero in the distribution is more than that of the higher values. Unlike in Uniform distribution where probability of choosing any value is equal.


***Model : CNN***
* The `Convolutional Neural Netowork` has 2 convolution layers and pooling layers with 3 fully connected layers. The first convolution layer takes in a channel of dimension 1 since the images are grayscaled. The kernel size is chosen to be of size 3x3 with stride of 1. The output of this convolution is set to 16 channels which means it will extract 16 feature maps using 16 kernels. We pad the image with a padding size of 1 so that the input and output dimensions are same. The output dimension at this layer will be 16 x 28 x 28. The we apply RelU activation to it followed by a max-pooling layer with kernel size of 2 and stride 2. This down-samples the feature maps to dimension of 16 x 14 x 14.
* The second convolution layer will have an input channel size of 16. We choose an output channel size to be 32 which means it will extract 32 feature maps. The kernel size for this layer is 3 with stride 1. We again use a padding size of 1 so that the input and output dimension remain the same. The output dimension at this layer will be 32 x 14 x 14. We then follow up it with a RelU activation and a max-pooling layer with kernel of size 2 and stride 2. This down-samples the feature maps to dimension of 32 x 7 x 7.
* Finally, 3 fully connected layers are used. We will pass a flattened version of the feature maps to the first fully connected layer. The fully connected layers have 1568 nodes at input layer, 512, 256 nodes in the first and second hidden layers respectively, with ouput layer of 10 nodes (10 classes). So we have two fully connected layers of size 1568 x 512 followed up by 512 x 256 and 256 x 10.
* The test accuracy is ***93.77%*** (***This result uses dropout probability of 20%***)
* A `convNet_model.pth` file has been included. With this one can directly load the model state_dict and use for testing.

<p align='center'>
  <img src='https://github.com/shrigulhane100/Handwritten-digits-classification-Tensorflow/blob/main/images/output.png'>
</p>


# Dependencies

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
- **Mac:** https://conda.io/projects/conda/en/latest/user-guide/install/macos.html
- **Windows:** https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/deep-learning-v2-pytorch.git
cd deep-learning-v2-pytorch
```

2. Create (and activate) a new environment, named `deep-learning` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n deep-learning python=3.6
	source activate deep-learning
	```
	- __Windows__: 
	```
	conda create --name deep-learning python=3.6
	activate deep-learning
	```
	
	At this point your command line should look something like: `(deep-learning) <User>:deep-learning-v2-pytorch <user>$`. The `(deep-learning)` indicates that your environment has been activated, and you can proceed with further package installations.

Sure! Here's the updated version that installs TensorFlow instead of PyTorch:

3. Install TensorFlow; this should install the latest version of TensorFlow.
- __Linux__, __Mac__, or __Windows__:
	```
	pip install tensorflow
	```

Is there anything else you need help with?

4. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

5. That's it!

Now most of the `deep-learning` libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project. 

Now, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd deep-learning-v2-pytorch
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.
