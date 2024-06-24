# 02466-Cell-Profiling



## Data description

The original folder contains cell images from the BBBC021 official website: https://bbbc.broadinstitute.org/BBBC021. The BBBC021 data set has 55 plates, every plate has 60 wells, and each well has 4 sites, so in total, there are 55x60x4=13200 sites. Each site has three channels, dapi, tubulin and actin, where the size in pixels are integer ranging from 0 to around 40000. These three channels can be concatenated as a RGB image, which corresponds to one multi-cell image. 3848 of the multi-cell images are labelled with mechanism of action (moa). In terms of folder structure, the original folder has 55 subfolder, and each subfolder corresponds to one plate. Each subfolder contains 720 tiff images, which corresponds to 60 well x 4 sites x 3 channel. Here is an example of BBBC021 dataset's naming rule. In the name 'Week3_290607_F10_s1_w191E8D91B-9023-490C-BDE0-B5475EA94DB5', 'F10' represents the well, 's1' represents the site, 'w1' represents the channel, and '91E8D91B-9023-490C-BDE0-B5475EA94DB5' is the unique identifier for the multi-image. Note 'Week3_290607' is not an accurate identifier of the plate. For detailed description, please refer to the website.

The singlecell folder consists of singh_cp_pipeline_singlecell_images folder and metadata.csv. The singlecell/singh_cp_pipeline_singlecell_images folder contains segmented single-cell images by Singh's CP pipeline from the original images labelling with moa. In this directory, there are 3848 subfolders, and each subfolder represents a multi-cell image, naming after the dapi channel (w1) of this multi-cell image. For example, the folder 'Week3_290607_F10_s1_w191E8D91B-9023-490C-BDE0-B5475EA94DB5' contains segmented single-cell images from multi-cell images. These single-cell images are named 'Week3_290607_F10_s1_w191E8D91B-9023-490C-BDE0-B5475EA94DB5_1.npy', 'Week3_290607_F10_s1_w191E8D91B-9023-490C-BDE0-B5475EA94DB5_2.npy' and so on. The number in the end of single-cell image name shows the index of single-cell images. In the singlecell/metadata.csv, each line represents a single-cell image, showing their corresponding multicell image (unsegmented cell image), compound, concentration moa. 

## Breakdown of helper files

**models.py** contains all the files used throughout the project

**plotting.py** contains all the plots used throughout the project

**loss_functions.py** contains all the loss functions used for the different models

**dataset_tools.py** contains helper classes and functions for loading and spliting the data into train, test and val split

## How to train a model
### VAE
Main file to train a model is **train.py** In the file there is a line:
```python
dataset = OwnDataset(transform=tf, path=r"C:\Users\Otto\Desktop\Fagprojekt_data\labelled_data")
```
This is the path to where the dataset is located. If no path is specified it will default to **data_subset**. This folder also showcases the format of how the data file should look

Further down we load the model and loss functions, and specify the latent dimension and hidden dimensions. In our case the hidden dimension is always 2 times the latent dimension
```python
from models import VAE_LAFARGE
from loss_functions import loss_function_mean as loss_function
model = VAE_LAFARGE(input_dim=(3,68,68), hidden_dim=512, latent_dim=256)
```

This is all you need to change to trian a different model, however you can also change hyperparameters such as weight decay, learning rate, optimizer etc.

## Autoencoder
THis is done in **autoencoder.ipynb**

### Latent Classifier
This is done in the **latent_classifier.ipynb** notebook. Same as before the models are loaded from **models.py** and trained on the given data

### Image classifier
THis is done in **independant_classifier.ipynb**

### KNN classifier
This is done in **KNN.ipynb**. Note that this requires you to hold the entire dataset in memory, making it not possible to run this fitting on most computers.

## Data analysis

### Exploratory analysis
Exploratory data analysis is done in **exploratory_analysis.ipynb**

### Data analysis of training data and comparison of models
This is done in **data_analysis.ipynb**

### McNemar test
Done in **Mcnemar_test.ipynb**


