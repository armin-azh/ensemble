# Ensemble Model Based on an Improved Convolutional Neural

This is a PyTorch/GPU implementation of the paper [Ensemble Model Based on an Improved Convolutional Neural](https://localhost.com):
```
@Article{EnsembleModelBasedOnAnImprovedConvolutionalNeural2022,
 Add the final citation
}
```


### Catalog
- [x] Download NSL-KDD and pretrains
- [x] Train
- [x] Test and ensemble to pretrain models


### Download NSL-KDD dataset
Download the NSL-KDD dataset on your local, run code bellow on project root
> ./download.sh

After that, you will see new **./Data** directory on project root.

### Train new classifiers
> python train.py --dir ./Data

### Test
#### Download Pretrains
> ./download_pretrains.sh
> 
Now there will be existed **./Pretrains** directory.

#### Evaluation
> python eval.py --pretrain ./Pretrains


### Licence