# SRMIST_23VIS47SRM_Pixel_shifter_DPD_image_using_generative_AI

## Overview

Monocular depth estimation is a computer vision task that aims to estimate the depth information of a scene using a single input image, typically an RGB image. The goal is to predict the distance from the camera to each pixel in the image, creating a depth map. This depth map can be used for various applications, such as 3D reconstruction, scene understanding, and augmented reality.
<br />
<br />
## Features
- Dataset handling: Organized NYU Depth V2 dataset into train, validation, and test sets.
- Data augmentation: Applied various techniques to augment training images for better generalization.
- Model architecture: Used a U-Net with ResNeXt-50-32x4d encoder for depth estimation.
- Training loop: Implemented training with MSE loss, AdamW optimizer, gradient clipping, and OneCycleLR scheduler.
- Evaluation metrics: Used SSIM and MSE to evaluate model performance on validation and test sets.
- Visualization: Displayed input images, ground truth, and predicted depth maps for qualitative assessment.
- Logging and saving: Logged training and validation metrics, saved the best model based on SSIM.
- Analysis: Analyzed metric relationships (e.g., SSIM vs. MSE) using heatmap plots for insights.<br /><br />

## Project Structure

The project is structured as follows:
- `monocular_Depth_Estimation_NYUV2.ipynb` : Final IPYNB file having highest accuracy model for depth estimation 
- `logs.csv` : Contain logs for the latest depth estimation file.
- `Experimental_changes` : Contains the experimental changes done to get the analysis from the depth estimation model.
    - Contains 9 files with changes leading to lower train time and higher results.
    - `Analyzer.ipynb` to show the best model from the 9 models present with analysis.<br /><br />

## Getting Started

### Prerequisites

- **Python**: Install Python, preferably version 3.6 or higher.
- **PyTorch**: Deep learning library for building and training neural networks. Install using `pip install torch`.
- **Albumentations**: Library for image augmentation. Install using `pip install albumentations`.
- **OpenCV**: Library for image and video processing. Install using `pip install opencv-python`.
- **Matplotlib**: Library for creating static, animated, and interactive visualizations. Install using `pip install matplotlib`.
- **Pandas**: Library for data manipulation and analysis. Install using `pip install pandas`.
- **Jupyter Notebook**: VS code extension for jupyter notebook.
- **Segmentation Models PyTorch**: Library with implementations of various segmentation models. Install using `pip install -Uq segmentation-models-pytorch`.
- **TQDM**: Library for adding progress bars to your loops. Install using `pip install tqdm`.
- **torchmetrics**: Library for metrics computation in PyTorch. Install using `pip install torchmetrics`.
- **Numpy**: For manipulation of vectors and array and its values. `pip install numpy`
- **Scikit image**: Python library for image processing that provides tools for transforming, filtering, segmenting, and measuring properties of images. `pip install scikit-image`<br />

### Usage

- **Kaggle API Key**: The instructions for using the Kaggle API key are accurate. However, in Google Colab, you need to upload the kaggle.json file manually using the file upload feature or through Google Drive mounting.
- **Running on Kaggle Notebook**: Running the code on Kaggle Notebook should work without any issues, as long as you have the necessary dataset (NYU Depth V2) added to your Kaggle dataset or uploaded it directly to the notebook.
- **Running on Local Machine**: Running the code on a local machine would require setting up the environment with the necessary libraries (torch, albumentations, opencv-python, matplotlib, pandas, tqdm, segmentation-models-pytorch, torchmetrics) and ensuring you have the kaggle.json file in the correct directory.
    
    - Ensure that all import statements are correct and libraries are installed.
    - Make sure that the file paths are correct and accessible.
    - Check that the dataset download and extraction paths are correct `'/content/nyu_data/data/nyu2_train.csv'`, `'/content/nyu_data/data/nyu2_train'`, `'/content/nyu_data'`.
    - Verify that the dataset loading and preprocessing steps are correct and handle any potential errors or inconsistencies.<br />

## How it works?

- ### Dataset Preparation
    - Download the NYU Depth V2 dataset using the Kaggle API and extract the files.
    - Organize the dataset into train, validation, and test sets.<br />
- ### Data Augmentation
    - Define Augmentation Techniques: Define a set of data augmentation techniques, such as horizontal flips, Gaussian noise, and color adjustments, to apply to the training images.<br />
- ### Model Architecture
    - U-Net with ResNeXt-50-32x4d Encoder: Use the U-Net model with a ResNeXt-50-32x4d encoder for depth estimation.
    - Model Initialization: Initialize the model and set the encoder to be trainable or frozen based on the training stage.<br />
    
    
        <img width="535" alt="Picture1" src="https://media.github.ecodesamsung.com/user/26108/files/dadf5e95-1ce6-4e4d-bd0b-d4d723ef972c">

    
- ### Training Loop
    - Loss Function and Optimizer: Use the Mean Squared Error (MSE) loss function and the AdamW optimizer for training.
    - Gradient Clipping: Implement gradient clipping to prevent exploding gradients.
    - Learning Rate Scheduler: Use a OneCycleLR scheduler for stable training.<br />
- ### Evaluation Metrics
    - Structural Similarity Index Measure (SSIM): Use SSIM to evaluate the similarity between the predicted and ground truth depth maps.
    - Mean Squared Error (MSE): Use MSE to measure the difference between the predicted and ground truth depth maps.<br />
- ### Training and Validation
    - Batch Iteration: Iterate over the training and validation datasets in batches.
    - Compute Loss and Metrics: Compute the loss and metrics (SSIM, MSE) for each batch, update the model weights, and log the metrics.<br />
- ### Logging and Saving
    - Logging: Log the training and validation metrics (loss, SSIM, MSE) for each epoch.
    - Model Saving: Save the best model based on the SSIM score on the validation set.<br />
- ### Testing
    - Prediction: Use the trained model to predict depth maps for the test dataset.
    - Evaluation Metrics: Compute the evaluation metrics (SSIM, MSE) for the test predictions.<br />
- ### Visualization
    - Qualitative Assessment: Visualize random samples of input images, ground truth depth maps, and predicted depth maps for qualitative assessment.<br />
- ### Analysis
    - Performance Analysis: Analyze the relationships between different metrics (e.g., SSIM vs. MSE) using heatmap plots to gain insights into the model's performance.
    - Please refer the `monocular_Depth_Estimation_NYUV2.ipynb.ipynb` for more analysis.<br />
- ### Results
    - we got a SSIM of 0.912 and MSE of 0.003<br /><br />
    
    
    
    <img width="827" alt="Picture2" src="https://media.github.ecodesamsung.com/user/26108/files/0143038b-b4f9-4afd-bf2f-b15689e2d142">

## Data Source

- https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2

## Screen Shots
   ### Data RGB image and ground truth depth image      
![ground](https://media.github.ecodesamsung.com/user/26108/files/3a48cd82-f5e4-476a-b459-12234b8a3391)
   ### Output after 5 epochs 
![output](https://media.github.ecodesamsung.com/user/26108/files/2ffc1ec6-69ae-4307-880b-3faac69070e9)
   ### Epoch vs Validation loss
![train](https://media.github.ecodesamsung.com/user/26108/files/c9681534-18c8-4ee3-b1f3-d36dc518615d)
   ### Final result and output
![result](https://media.github.ecodesamsung.com/user/26108/files/78199c7b-c493-4e33-91bd-0d75bddae70a)




