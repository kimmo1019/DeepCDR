import argparse
import random,os,sys
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from keras.utils import multi_gpu_model,plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse
####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, help='use gene expression or not')
parser.add_argument('-use_cn', dest='use_cn', type=bool, help='use copy number or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut,use_gexp,use_cn,use_methy = args.use_mut,args.use_gexp, args.use_cn, args.use_methy
israndom=args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+('with_cn' if use_cn else 'without_cn')+'_'+('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' +GCN_deploy
#symb = sys.argv[2]#0,1,2,3,4 --> master, fc, shallow,tanh, unified
####################################Constants Settings###########################
TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
DPATH = '/home/liuqiao/software/Drug_response/data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
#Drug_feature_file = '%s/GDSC/pubchem_descriptor_3072features_fingerprints.csv'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Cell_line_feature_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
Copy_number_file = '%s/CCLE/genomic_copynumber_561celllines_710genes_demap_features.csv'%DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
Max_atoms = 100


def MetadataGenerate(Drug_info_file,Cell_line_info_file,Cell_line_feature_file,Drug_feature_file,Gene_expression_file,Copy_number_file,Methylation_file,filtered):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    #load demap cell lines genomic mutation features
    cell_line_feature = pd.read_csv(Cell_line_feature_file,sep=',',header=0,index_col=[0])
    cell_line_id_set = list(cell_line_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    #only keep overlapped cell lines
    cell_line_feature = cell_line_feature.loc[list(gexpr_feature.index)]
    
    #load gene copy number file
    gene_cn_feature = pd.read_csv(Copy_number_file,sep=',',header=0,index_col=[0])
    
    #load methylation 
    methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
    assert methylation_feature.shape[0]==gene_cn_feature.shape[0]==gexpr_feature.shape[0]==cell_line_feature.shape[0]        
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in cell_line_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    data_idx.append((each_cellline,pubchem_id,ln_IC50,cellline2cancertype[each_cellline])) 
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return cell_line_feature, drug_feature,gexpr_feature,gene_cn_feature,methylation_feature, data_idx
    
def DataSplit(data_idx,ratio = 0.95):#leave drug out
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx
#could speed up using multiprocess and map  
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def FeatureExtract(data_idx,drug_feature,cell_line_feature,gexpr_feature,gene_cn_feature,methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    #nb_drug_features = drug_feature.shape[1]
    nb_cell_line_features = cell_line_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_cn_features = gene_cn_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    #modify
    drug_data = [[] for item in range(nb_instance)]
    cell_line_data = np.zeros((nb_instance,1, nb_cell_line_features,1),dtype='float32')
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    gene_cn_data = np.zeros((nb_instance, nb_cn_features),dtype='float32') 
    methylation_data = np.zeros((nb_instance, nb_methylation_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,ln_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        cell_line_data[idx,0,:,0] = cell_line_feature.loc[cell_line_id].values
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        gene_cn_data[idx,:] = gene_cn_feature.loc[cell_line_id].values
        methylation_data[idx,:] = methylation_feature.loc[cell_line_id].values
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,cell_line_data,gexpr_data,gene_cn_data,methylation_data,target,cancer_type_list
    


class MyCallback(Callback):
    def __init__(self,validation_data,patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save('../checkpoint/MyBestDeepCDR_%s.h5'%model_suffix)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
        print 'pcc-val: %s' % str(round(pcc_val,4))
        if pcc_val > self.best:
            self.best = pcc_val
            self.wait = 0
            self.best_weight = self.model.get_weights()
        else:
            self.wait+=1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        return
        



def ModelTraining(model,X_drug_data_train,X_cell_line_data_train,X_gexpr_data_train,X_gene_cn_data_train,X_methylation_data_train,Y_train,validation_data,nb_epoch=100):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='mean_squared_error',metrics=['mse'])
    #EarlyStopping(monitor='val_loss',patience=5)
    callbacks = [ModelCheckpoint('../checkpoint/best_DeepCDR_%s.h5'%model_suffix,monitor='val_loss',save_best_only=False, save_weights_only=False),
                MyCallback(validation_data=validation_data,patience=10)]
    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
    #validation data
    model.fit(x=[X_drug_feat_data_train,X_drug_adj_data_train,X_cell_line_data_train,X_gexpr_data_train,X_gene_cn_data_train,X_methylation_data_train],y=Y_train,batch_size=64,epochs=nb_epoch,validation_split=0,callbacks=callbacks)
    return model


def ModelEvaluate(model,X_drug_data_test,X_cell_line_data_test,X_gexpr_data_test,X_gene_cn_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,file_path):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_cell_line_data_test,X_gexpr_data_test,X_gene_cn_data_test,X_methylation_data_test])
    overall_pcc = pearsonr(Y_pred[:,0],Y_test)[0]
    print("The overall Pearson's correlation is %.4f."%overall_pcc)
    #regard cancer type
    f_out = open(file_path,'w')
    cancertype2pcc = {}
    for each in TCGA_label_set:
        ind = [b for a,b in zip(cancer_type_test_list,list(range(len(cancer_type_test_list)))) if a[0]==each]
        if len(ind)>1:
            pcc = pearsonr(Y_pred[:,0][ind],Y_test[ind])[0]
            each = each.replace('/','-')
            cancertype2pcc[each] = pearsonr(Y_pred[:,0][ind],Y_test[ind])[0]
            hkl.dump([Y_pred[:,0][ind],Y_test[ind]],'%s/outcome/DeepCDR_cancer_type_%s_%.4f_%s.hkl'%(DPATH,each,pcc,model_suffix))
            f_out.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2pcc[each]))
    f_out.write("AvegePCC\t%.4f.\n"%overall_pcc)
    f_out.close()
    
    #regard durg
    drug_list = list(set([item[-1] for item in cancer_type_test_list]))
    for each in drug_list:
        ind = [b for a,b in zip(cancer_type_test_list,list(range(len(cancer_type_test_list)))) if a[-1]==each]
        pcc = pearsonr(Y_pred[:,0][ind],Y_test[ind])[0]
        hkl.dump([Y_pred[:,0][ind],Y_test[ind]],'%s/outcome/DeepCDR_drug_%s_%.4f_%s.hkl'%(DPATH,each,pcc,model_suffix))
    return cancertype2pcc

def main():
    random.seed(0)
    #model = KerasModel().createUnified(3072,28328)
    #Meta data loading
    cell_line_feature, drug_feature,gexpr_feature,gene_cn_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Cell_line_feature_file,Drug_feature_file,Gene_expression_file,Copy_number_file,Methylation_file,False)
    #print cell_line_feature.shape,drug_feature.shape,gexpr_feature.shape,gene_cn_feature.shape,methylation_feature.shape
    #sys.exit()
    #Meta data split
    data_train_idx,data_test_idx = DataSplit(data_idx)
    #Extract features for training and test 
    X_drug_data_train,X_cell_line_data_train,X_gexpr_data_train,X_gene_cn_data_train,X_methylation_data_train,Y_train,cancer_type_train_list = FeatureExtract(data_train_idx,drug_feature,cell_line_feature,gexpr_feature,gene_cn_feature,methylation_feature)
    X_drug_data_test,X_cell_line_data_test,X_gexpr_data_test,X_gene_cn_data_test,X_methylation_data_test,Y_test,cancer_type_test_list = FeatureExtract(data_test_idx,drug_feature,cell_line_feature,gexpr_feature,gene_cn_feature,methylation_feature)
    #print np.isnan(X_gene_cn_data_test).sum()
    #sys.exit()

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    
    validation_data = [[X_drug_feat_data_test,X_drug_adj_data_test,X_cell_line_data_test,X_gexpr_data_test,X_gene_cn_data_test,X_methylation_data_test],Y_test]
    #test not randomlize which will introduce bias
    # whether randomlize feature
    #if not use_gexp:
    #    X_gexpr_data_train = np.random.rand(X_gexpr_data_train.shape[0],X_gexpr_data_train.shape[1])
    #if not use_cn:
    #    X_gene_cn_data_train = np.random.rand(X_gene_cn_data_train.shape[0],X_gene_cn_data_train.shape[1])
    #if not use_methy:
    #    X_methylation_data_train = np.random.rand(X_methylation_data_train.shape[0],X_methylation_data_train.shape[1])  
    #modify model
    #model = KerasMultiSourceModel().createMaster(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2],X_gexpr_data_train.shape[-1],X_gene_cn_data_train.shape[-1],X_methylation_data_train.shape[-1])
    print X_drug_data_train[0][0].shape[-1],X_cell_line_data_train.shape[-2],X_gexpr_data_train.shape[-1],X_gene_cn_data_train.shape[-1],X_methylation_data_train.shape[-1]
    model = KerasMultiSourceGCNModel(use_mut,use_gexp,use_cn,use_methy).createMaster(X_drug_data_train[0][0].shape[-1],X_cell_line_data_train.shape[-2],X_gexpr_data_train.shape[-1],X_gene_cn_data_train.shape[-1],X_methylation_data_train.shape[-1],args.unit_list,args.use_relu,args.use_bn,args.use_GMP)


    #construct five models
#     if symb == '0':
#         model = KerasModel().createMaster(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
#     elif symb == '1':
#         model = KerasModel().createFC(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
#     elif symb == '2':
#         model = KerasModel().createShallow(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
#     elif symb == '3':
#         model = KerasModel().createTanh(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
#     elif symb == '4':
#         model = KerasModel().createUnified(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
#     else:
#         print 'Please input 0-4 for model selection\n'
#         sys.exit(1)
        
    #model = ModelConstruct(X_drug_data_train.shape[-2],X_cell_line_data_train.shape[-2])
    #model.summary()
    print('Begin training...')
    model = ModelTraining(model,X_drug_data_train,X_cell_line_data_train,X_gexpr_data_train,X_gene_cn_data_train,X_methylation_data_train,Y_train,validation_data,nb_epoch=100)

    print('Training done')
    cancertype2pcc = ModelEvaluate(model,X_drug_data_test,X_cell_line_data_test,X_gexpr_data_test,X_gene_cn_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,'%s/DeepCDR_%s.log'%(DPATH,model_suffix))
    print('Evaluation finished!')

if __name__=='__main__':
    main()
    
