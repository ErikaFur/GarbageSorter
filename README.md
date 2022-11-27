# GarbageSorter

We propose a DenseNet-based deep-learning computer vision model for classifying garbage into multiple classes (cardboard/paper, glass, metal and plastic) using a SCARA
manipulator which has four degrees of freedom. The idea is the following: an image of garbage is taken as input from the camera, and then the model will have to give a result in a form of a garbage class number - to which of the described garbage classes this object belongs. The output will be passed as an input to the Raspberry Pi and based on this number, the SCARA manipulator will throw garbage into one of the containers in accordance with their markings.


### How to use classification model?

1. Install requirements from GarbageSorter/requirements.txt
2. Run GarbageSorter/main.ipynb from GarbageSorter directory (it contains relative pathes)
