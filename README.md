# DeepCDR
Cancer Drug Response Prediction via a Hybrid Graph Convolutional Network
 
 ![model](https://github.com/kimmo1019/DeepCDR/blob/master/model.png)
 
 DeepCDR is a hybrid graph convolutional network for cancer drug response prediction.
 
 # Requirements
- Keras==2.1.4
- TensorFlow==1.13.1
- hickle >= 2.1.0

# Installation
DeepCDR can be downloaded by
```shell
git clone https://github.com/kimmo1019/DeepCDR
```
Installation has been tested in a CentOS/MacOS platform.

# Instructions
We provide detailed step-by-step instructions for running DeepCDR model including data preprocessing, model training, and model test.

## Data preprocessing
**Step 1**: Generating genomic mutation matrix, gene expression matrix and DNA methylation matrix

```python
python DataPreprocess.py [drug screening file] [genomic mutation file] [gene expression file] [DNA methylation file]
[drug screening file] - Drug information and IC50 values from GDSC database
[genomic mutation file] - Genomic profile from CCLE database
[gene expression file] - Transcritomic profile from CCLE database
[DNA methylation file] - Epigenomic profile from CCLE database
```
The preprocessed data will be in `data` folder.

**Step 2**: DeepCDR model training and testing

```python
python run_DeepCDR.py -gpu_id [gpu_id] -use_mut [use_mut] -use_expr [use_gexp] -use_methy [use_methy] -checkpoint_dir [checkpoint_dir]
[gpu_id] - set GPU card id (default:0)
[use_mut] - whether use genomic mutation data (default: True)
[use_gexp] - whether use gene expression data (default: True)
[use_mut] - whether use DNA methylation data (default: True)
[log_dir] - location to save the trained model
```

The prediction outcome will be saved in `data` folder.


# License
This project is licensed under the MIT License - see the LICENSE.md file for details


























