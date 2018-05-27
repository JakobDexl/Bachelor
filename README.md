# Bachelor Draft

![Learning](https://github.com/JakobDexl/Bachelor/blob/master/Test_visulizations/stack2.gif) <br />
*Something nice looking is to make a activaton.gif <br />
during the training process*

## Purpose
The number of prescribed CT and MR admissions is continuously increasing [Statista](https://github.com/JakobDexl/Bachelor/blob/master/Test_visulizations/statistic_id172719_ct-und-mrt---untersuchungszahlen.png). This often leaves radiologists with little time per scan. In addition, internal studies have shown that most of the findings are normal. This correlation suggests to support physicians with a pre-classification tool. 
Such a tool that assists the Radiologist in classifying brain MRI images into normal and abnormal should be transparent and be able to justify its answers. This is necessary to meet the high standards of medical technologies.
State of the art results in this area could be achieved through good feature engineering and machine learning models ().
This requires a lot of mathematical and medical knowledge. In the last few years deep learning models have become popular. These won numerous classification competitions and beat classic feature engineering (reference). These models automatically extract features based on a data set. Recent papers try to use this potential for medicine as well (ref).  
Problems that occur here are the low amount of data available. Furthermore, the high heterogeneity of this data and the loss of transparency due to the low control of feature extraction.

## Goals

I try to understand whats going on in a CNN model for Brain Mri images (and to use this informaton to make better models) <br />
I try out visulization techniques and make them easy accesible for keras models <br />
Finally i try to use natural Information (kind of that way the radiologist look at the pictures) for tuning models <br />

## Common ways to investigate a model

(1) - Visualize filters <br />
(2) - Visualize activationmaps <br />
(3) - Occluding experiments <br />
(4) - Attentionmaps <br />
(5) - Deconvolution <br />

## Refereces
### Paper

- [Selvaraju, Ramprasaath R.; Cogswell, Michael; Das, Abhishek; Vedantam, Ramakrishna; Parikh,
Devi; Batra, Dhruv (2017): Grad-CAM. Visual Explanations from Deep Networks via Gradient-
based Localization, 21.03.2017](http://arxiv.org/pdf/1610.02391) (4) <br />
- [Yosinski, Jason; Clune, Jeff; Nguyen, Anh; Fuchs, Thomas; Lipson, Hod (2015): Understanding
Neural Networks Through Deep Visualization, 22.06.2015](http://arxiv.org/pdf/1506.06579) (1,2,4,5) <br />
- [Zeiler, Matthew D.; Fergus, Rob (2013): Visualizing and Understanding Convolutional
Networks, 28.11.2013](http://arxiv.org/pdf/1311.2901) (5) <br />
- [Zintgraf, Luisa M.; Cohen, Taco S.; Adel, Tameem; Welling, Max (2017): Visualizing Deep
Neural Network Decisions. Prediction Difference Analysis, 15.02.2017](http://arxiv.org/pdf/1702.04595) (4) <br />

### Blogs

- [Understanding and Visualizing Convolutional Neural Networks ](http://cs231n.github.io/understanding-cnn/) (1,2,3,4,5) <br />
- [Feature Visualization - How neural networks build up their understanding of images](https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis) (1,2) <br />
- [Deepvis](http://yosinski.com/deepvis) (1,2,4,5) <br />
- [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) (2, 4) <br />
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) (5) <br />

### Misc

- [Data Preprocessing](http://cs231n.github.io/neural-networks-2/) <br />



