
## Requirements
Our experiments were trained and tested on:
+ Python 3.8
+ PyTorch 2.3.0
+ scikit-learn 0.23.2
+ numpy 1.23.1
## Datasets
You will need to download the datasets yourself. For DAGM Dataset you will also need a Kaggle account.
* DAGM available [here.](https://www.kaggle.com/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
* KolektorSDD2 available [here.](https://www.vicos.si/Downloads/KolektorSDD2)
## Training and Testing
* Train on DAGM and KSDD2 with 3 GPUs
```Shell
./DEMO.sh 0 1 2
```
Each experiment is run on a single GPU. Please modify the `DEMO.sh` script according to the number of GPUs you have available.

## Acknowledgement
* The code is borrowed from [Mixed supervision for surface-defect detection: from weakly to fully supervised learning](https://github.com/vicoslab/mixed-segdec-net-comind2021)
## Citation

Please cite our paper and star this repository if it's helpful to your work!
