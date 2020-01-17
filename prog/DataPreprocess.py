'''
Usage:
python DataPreprocess.py [drug screening file] [genomic mutation file] [gene expression file] [DNA methylation file]
[drug info file] - Drug information including name, pubchem id, etc
[drug screening file] - Drug information and IC50 values from GDSC database
[genomic mutation file] - Genomic profile from CCLE database
[gene expression file] - Transcritomic profile from CCLE database
[DNA methylation file] - Epigenomic profile from CCLE database
This script is for generating genomic mutation matrix, gene expression matrix and DNA methylation matrix from raw data in CCLE database.
'''
import csv
import pandas as pd
import numpy as np
#input raw file list (from GDSC database or CCLE/Demap database)
'''
GDSC_drug_file = 'data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Mutation_file = 'data/CCLE/CCLE_mutations.csv'
Experiment_file = 'data/CCLE/GDSC_IC50.csv'
Expression_raw_file = 'data/CCLE/CCLE_expression.csv'#coding genes
TSS_methylation_raw_file = 'data/CCLE/CCLE_RRBS_TSS_1kb_20180614.txt'
'''
GDSC_drug_file = sys.argv[1]
Experiment_file = sys.argv[2]
Mutation_file = sys.argv[3]
Expression_raw_file = sys.argv[4]
TSS_methylation_raw_file = sys.argv[5]



#load drug file GDSC ID to pubchem id 

csv_reader = csv.reader(open(GDSC_drug_file,'r'))
rows = [each for each in csv_reader]
drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

def MutationProcess(Mutation_file):
    cellline_id_set = []
    mutation_list = []
    for line in open(Mutation_file).readlines()[1:]:
        chrom = line.split(',')[5]
        start = line.split(',')[6]
        gene_name = line.split(',')[2]
        cellline_id = line.strip().split(',')[-1]
        if gene_name in frequently_mutated_genes:
            cellline_id_set.append(cellline_id)
            content = gene_name+'.'+chrom+':'+start
            mutation_list.append(content)
    return cellline_id_set,mutation_list
cellline_id_set,mutation_list = MutationProcess(Mutation_file)
cellline_id_set_uniq = list(set(cellline_id_set))
mutation_list_uniq = list(set(mutation_list))
experiment_data = pd.read_csv(Experiment_file,sep=',',header=0,index_col=[0])
drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
experiment_data_filtered = experiment_data.loc[drug_match_list]
meta_data_list=[]
for each_drug in experiment_data_filtered.index:
    for each_cellline in experiment_data_filtered.columns:
        if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]):
            meta_data_list.append([each_drug,each_cellline,float(experiment_data_filtered.loc[each_drug,each_cellline])])
cellline_match_list=[]
meta_data_cellline_list_uniq = list(set([item[1] for item in meta_data_list]))
for each in cellline_id_set_uniq:
    if each in meta_data_cellline_list_uniq:
        cellline_match_list.append(each)
cellline_id_set_filtered=[]
mutation_list_filtered=[]
for line in open(Mutation_file).readlines()[1:]:
    chrom = line.split(',')[5]
    if chrom=='X':
        chrom='23'
    if chrom=='Y':
        chrom='24'
    start = line.split(',')[6]
    gene_name = line.split(',')[2]
    cellline_id = line.strip().split(',')[-1]
    if gene_name in frequently_mutated_genes and cellline_id in cellline_match_list:
        cellline_id_set_filtered.append(cellline_id)
        content = gene_name+'.'+chrom+':'+start
        mutation_list_filtered.append(content)
cellline_id_set_filtered_uniq = list(set(cellline_id_set_filtered))
mutation_list_filtered_uniq = list(set(mutation_list_filtered))
mutation_list_filtered_uniq.sort(key=lambda a : (int(a.split(':')[0].split('.')[-1]),int(a.split(':')[-1])))
feature_mat = np.zeros((len(cellline_id_set_filtered_uniq),len(mutation_list_filtered_uniq)),dtype = 'int32')
for line in open(Mutation_file).readlines()[1:]:
    chrom = line.split(',')[5]
    if chrom=='X':
        chrom='23'
    if chrom=='Y':
        chrom='24'
    start = line.split(',')[6]
    gene_name = line.split(',')[2]
    cellline_id = line.strip().split(',')[-1]
    if gene_name in frequently_mutated_genes and cellline_id in cellline_id_set_filtered_uniq:
        content = gene_name+'.'+chrom+':'+start
        col_idx = mutation_list_filtered_uniq.index(content)
        row_idx = cellline_id_set_filtered_uniq.index(cellline_id)
        feature_mat[row_idx][col_idx] = 1
df = pd.DataFrame(feature_mat, index=cellline_id_set_filtered_uniq, columns=mutation_list_filtered_uniq)
df.to_csv('../data/CCLE/genomic_mutation_34673_demap_features.csv')



