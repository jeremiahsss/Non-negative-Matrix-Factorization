# Non-negative-Matrix-Factorization

## Project Description
This study aims to assess the robustness of various Non-negative Matrix Factorization (NMF) algorithms when applied to datasets contaminated by high levels of noise or corruption. NMF is a valuable tool for feature extraction and dimensionality reduction in high-dimensional data analysis. However, the original NMF, which minimizes the squared error function, exhibits challenges in effectively handling noisy data and outliers. To overcome this limitation, multiple NMF variants have been developed, incorporating different objective functions and regularization techniques to enhance robustness. In this study, we comprehensively evaluated the robustness of four NMF algorithms: L1-NMF, L2-NMF, Huber-NMF, and CIM-NMF, in the presence of three types of noise contamination, namely block occlusion, salt and pepper, and Gaussian noise. Our experiment utilized two realworld facial image datasets, ORL and Extended Yale B. This report first provides a detailed exploration of the mathematical concepts and optimization approaches underlying each NMF variant and noise, followed by a meticulous analysis of the experimental results. Our experiment shows that all NMF variants outperform the original L2-NMF across different noise types. CIM-NMF and Huber-NMF exhibit stable and robust performance, with CIM-NMF excelling in handling salt and pepper noise. L1-NMF performs exceptionally well with Gaussian noise and low noise level scenarios but becomes unstable as noise levels increase.

## How to run the project
1. Unzip the datasets
2. Run the experiment.ipynb 

## Dataset Description
This study utilized two real-world facial image datasets, namely ORL and Extended Yale B datasets. Detailed descriptions of the datasets are as follows:

The ORL dataset comprises a collection of 400 facial images, featuring 40 unique subjects. These images were captured under varying conditions, including different lighting scenarios, facial expressions, and facial details. Notably, all images were taken against a uniformly dark background, with subjects in an upright, frontal position. In our experiment, to facilitate uniform analysis and reduce computation complexity, all images were resized to a standardized dimension of 37×30 pixels.

The Extended YaleB dataset includes a total of 2414 images, featuring 38 unique subjects. The images were captured under nine different poses and across 64 distinct illumination conditions. Prior to inclusion in the dataset, all images underwent manual alignment and cropping. In our experiment, to ensure consistency and reduce computation complexity, all images were resized to a standardized dimension of 48x42 pixels.
