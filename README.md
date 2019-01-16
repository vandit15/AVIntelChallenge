# Intel Analytics Vidhya Scene Classification Challenge
Pytorch implementation for the scene classification challenge.

### Hardware used
Google Colab GPU - Nvidia K80 - 12 GB memory

### Software Requirements
* Python 3.6
* Pytorch with GPU support
* skimage
* matplotlib
* opencv
* sklearn
* tqdm

## Solution Description
Following pretrained imagenet models were used:
* Resnet50
* Resnet101
* Densenet121
* Squeezenet1_1 (Took only ~ 1.5 GB of GPU space. So quite memory efficient.)

### Hyperparameters
* LR - 1e-3
* Batch-size - 32
* Validation split - .15
* All models were trained for 30 epochs.
 
### Ensembles
Simple voting procedure done among output from Resnet50, Resnet101 and Densenet121 resulted in the best testing accuracy of **94.33**.
 
We also tried stacking method but that surprisingly that resulted in drop in accuray to **92%**.  
 
## Trained models
Trained models can be found at [this](https://drive.google.com/file/d/17Ggkkzi7dhHCbVj6773xI1AN82ydlD1B/view?usp=sharing) drive link

### Next Steps
* Improving hyperparameter tuning.
* Fiddling with transforms (normalisation etc.)
* More complex ensemble methods
