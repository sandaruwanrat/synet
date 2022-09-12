# synet


* This project aimed to create CNN + LSTM models for 6 ranscription factors


# Requirements
    * keras==2.8.0
    * keras-preprocessing==1.1.2
    * tensorflow==2.8.0
    * scipy=1.7.3
    * numpy=1.21.2


# setup environment
* The conda env can be created using **synet.yml** file 


## Folder structure

* SOY_models
  * WRKY
  * RAV
* ARB_models
  * NAC
  * MYB
  * bHLH
  * ERF

* Each of the the TF directory has 3 directores.
  * Train
  Consist of training scripts and data.
  * Models
  The models are saved here by the script. However, you may have to update the paths in the training script.
  * Prediction
  Consist of prediction scripts 
