# Bachelor Draft

![Learning](https://github.com/JakobDexl/Bachelor/blob/master/Test_visulizations/stack2.gif) <br />
*the animation above shows the progression of the <br />
activation maps of the first layer during training*

## Purpose

The number of prescribed CT and MR admissions is continuously increasing [[Statista]](https://github.com/JakobDexl/Bachelor/blob/master/Test_visulizations/statistic_id172719_ct-und-mrt---untersuchungszahlen.png). This often leaves radiologists with little time per scan. In addition, internal studies have shown that most of the findings are normal. This correlation suggests to support physicians with a pre-classification tool. 
Such a tool that assists the Radiologist in classifying brain MRI images into normal and abnormal should be transparent and be able to justify its answers. This is necessary to meet the high standards of medical technologies.
State of the art results in this area could be achieved through good feature engineering and machine learning models (Sarita et al. 2013).
This requires a lot of mathematical and medical knowledge. In the last few years deep learning models have become popular. These won numerous classification competitions and beat classic feature engineering (reference). These models automatically extract features based on a data set. Recent papers try to use this potential for medicine as well (Mohsen et al. 2017; Rezaei et al. 2017).  
Problems that occur here are the low amount of data available. Furthermore, the high heterogeneity of this data and the loss of transparency due to the low control of feature extraction.

## Goals

- CNN models should be investigated and visualized during classification of brain MRI images. For this purpose a simple accessible api will be developed for keras models.<br />
- More controlled knowledge should be put back into the models.<br /> 
- Finally the human classification process (the way the radiologist looks at the pictures) should be digitally mimic and transferred to models.<br />

These objectives create more transparency and justify classification results.

## Theory
### Pathologies
#### Tumors
#### Neurodegenerative diseases
#### Vascular diseases 
#### Trauma
#### Multiple sclerosis
#### Infections
### Natural learning
### Machine learning
### CNN
### Visualization methods
#### Common ways to investigate a model

(1) - Visualize filters <br />
(2) - Visualize activationmaps <br />
(3) - Occluding experiments <br />
(4) - Attentionmaps (CAM, grad-Cam) <br />
(5) - Deconvolution <br />


## Method
### Data
### Model
### Visualization
### Knowledge transfer
### Tests
### 3D scaling
## Results
## Discussion
## Conclusion
## Refereces
### Paper

- [El-Dahshan, El-Sayed A.; Mohsen, Heba M.; Revett, Kenneth; Salem, Abdel-Badeeh M. (2014): Computer-aided diagnosis of human brain tumor through MRI. A survey and a new algorithm. In: Expert Systems with Applications 41 (11), S. 5526–5545. DOI: 10.1016/j.eswa.2014.01.021.](http://dx.doi.org/10.1016/j.eswa.2014.01.021)
- [Mohsen, Heba; El-Dahshan, El-Sayed A.; El-Horbaty, El-Sayed M.; Salem, Abdel-Badeeh M. (2017): Classification using deep learning neural networks for brain tumors. In: Future Computing and Informatics Journal.DOI: 10.1016/j.fcij.2017.12.001.](http://dx.doi.org/10.1016/j.fcij.2017.12.001)
- [Rezaei, Mina; Yang, Haojin; Meinel, Christoph (2017): Deep Learning for Medical Image Analysis, 17.08.2017](http://arxiv.org/pdf/1708.08987)
- [Saritha, M.; Paul Joseph, K.; Mathew, Abraham T.(2013): Classification of MRI brain images using combined wavelet entropy based spider web plots and probabilistic neural network, 2013.08.017. DOI: 10.1016/j.patrec.2013.08.017.](https://doi.org/10.1016/j.patrec.2013.08.017)
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



