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
**Step 1**: Download raw DNase-seq and RNA-seq data

We provided `1.Download_raw_data.sh` for download RNA-seq data (.tsv) and DNase-seq data (.narrowPeak and .bam) from the ENCODE project
We pre-defined cell type ID from 1-55. After downloading the meta data from ENCODE website (`head -n 1 files.txt|xargs -L 1 curl -O -L`), one can run the following script:

```python
bash 1.Download_raw_data.bash  -c <CELL_ID> -r -p -b
-c  CELLID: pre-defined cell ID (from 1 to 55)
-r  download RNA-seq data (.tsv)
-p  download chromatin accessible peaks from DNase-seq data (.narrowPeak)
-b  download chromatin accessible readscount from DNase-seq data (.bam)
```
one can also run ```bash 1.Download_raw_data.bash  -h``` to show the script instructions. Note that `.bam` files downloading may take time. After downloading the raw data, the raw data folder will be organized by `cell-assay-experiment-file` order. Note that each experiment may contain multiple replicates. See an example of the folder tree:

```
data/
    |-- raw_data/
    |   |-- 1/
    |   |   |-- dseq/
    |   |   |   |-- ENCSR000EIE/
    |   |   |   |   |-- ENCFF953HEA.bed.gz
    |   |   |   |   |-- ENCFF983PML.bam
    |   |   |   |-- ENCSR000ELW/
    |   |   |   |   |...
    |   |   |-- rseq/
    |   |   |   |-- ENCSR000BXY/
    |   |   |   |   |-- ENCFF110IED.tsv
    |   |   |   |   |-- ENCFF219FVQ.tsv
    |   |   |   |-- ENCSR000BYH/
    |   |   |   |   |...
```

**Step 2**: Merge multiple replicates of DNase-seq and RNA-seq data

We merge multiple replicate of RNA-seq data by taking the average expression of each gene across replicates in a cell type. As for DNase-seq data, we only keep bins that appear in more than half of the replicates with respect to a cell type. One can run the following scripts to merge relicates of both DNase-seq and RNA-seq data. Note that the referece genome (`hg19`) will be automatically downloaded.

```python
python 2.Merge_multi_rep_data  <CELL_ID> 
CELLID: pre-defined cell ID (from 1 to 55)
```
The merged data (`e.g. 1.TPM.tsv and 1.peak.bins.bed`) will be located in `data/processed_RNA_DNase` folder.

**Step 3**: Loci filtering and candidate regulatory regions selection

Please refer to `Supplementary Figure 1` for candidate regulatory regions selection strategy. Directly run `bash 3.0.Generate_peak_bin.sh` to generate candidate regulatory regions set (`union.peaks.bed` and `union.peaks.pad1k.bed`)

**Step 4**: Generating expression matrix (N x C)

The TF gene expression matrix size is `N x C` where N is the number of TFs and C is the number of cell lines. 

```python
python 3.1.Generate_tf_exp.py <CELL_SET> <OUTPUT>
CELL_SET: cell id set
OUTPUT: output expression matrix file
```
**Step 5**: Generating motif score matrix (L x N)

The motif score matrix size is `L x N` where L is the number of candidate regulatory loci and N is the number of the coresponding TFs.

```python
python 3.2.Generate_motif_score.py <PEAK_FILE> <MOTIF_FILE> <OUTPUT>
PEAK_FILE: the generated union peak file in `Step 3` (e.g. `union.peaks.bed`)
MOTIF_FILE: motif file in homer format
OUTPUT: output motif score matrix file
```
**Step 6**: Generating label matrix (L x C)

We provide scripts for generating both binary label matrix (classification) and continuous label matrix (regression) here.

The label matrix size is `L X C` where L is the number of candidate regulatory loci and C is the number of cell lines.

Use the following two scripts for generating binary label matrix
```python
python 3.3.Generate_label.py <PEAK_FILE> <CELL_SET> <OUTPUT> / 3.4.Generate_label.py <PEAK_FILE> <CELL_SET> <OUTPUT>
PEAK_FILE: the generated union peak file in `Step 3` (e.g. `union.peaks.bed`)
CELL_SET: cell id set
OUTPUT: output label matrix file
```
**Step 7**: Normalizing reads count

For reads count across different cell line, we normalize it by log transformation.
```python
python 3.5.Normalize_readscount.py <CELL_SET> <OUTPUT>
CELL_SET: cell id set
OUTPUT: output normalized reads count matrix file
```
**NOTES**: If one need to run DeepCAGE with custom data, what he/she needs to do is to generate three matrices (`TF expression matrix`, `motif score matrix` and `label matrix`) by own. 

## Model training and test

We provide `4.classification.py` and `5.Regression.py` for run DeepCAGE in a classication and regression settings, respectively.
```python
python 4.classification.py <GPU_ID> <FOLD_ID>
GPU_ID: GPU card id, default: 0
FOLD_ID: cross validation fold id, from 0-4
```
```python
python 5.Regression.py <GPU_ID> <FOLD_ID>
GPU_ID: GPU card id, default: 0
FOLD_ID: cross validation fold id, from 0-4
```
The model will be saved in `data/models` folder and prediction outcome will be saved in `data` folder.


# License
This project is licensed under the MIT License - see the LICENSE.md file for details


























