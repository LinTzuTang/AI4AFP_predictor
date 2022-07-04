from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,accuracy_score,f1_score,matthews_corrcoef,confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import numpy as np
import os

def split(data, labels, test_size = 0.1, random_state = 10, save = True, output_root = None):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size, random_state, stratify = labels)
    if save:
        output_root = output_root or os.getcwd()
        if not os.path.isdir(output_root):
            os.mkdir(output_root)
        json.dump(train_data.tolist(), open(os.path.join(output_root,'train_data.json'), "w"), indent=4) 
        json.dump(test_data.tolist(), open(os.path.join(output_root,'test_data.json'), "w"), indent=4) 
        json.dump(train_labels.tolist(), open(os.path.join(output_root,'train_labels.json'), "w"), indent=4) 
        json.dump(test_labels.tolist(), open(os.path.join(output_root,'test_labels.json'), "w"), indent=4) 
        print('train test encoded data and labels saved')
    return  train_data, test_data, train_labels, test_labels

def learning_curve(history, save = False, output_path_name = None):
    fig1 = plt.figure(figsize=(15,5))
    gs = gridspec.GridSpec(1, 2) 
    ax1 = fig1.add_subplot(gs[0,0])
    ax2 = fig1.add_subplot(gs[0,1])

    ax1.set_title('Train Accuracy',fontsize = '14' )
    ax2.set_title('Train Loss', fontsize = '14' )
    ax1.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax1.set_ylabel('Acc', fontfamily = 'serif', fontsize = '13' )
    ax2.set_xlabel('Epoch', fontfamily = 'serif', fontsize = '13' )
    ax2.set_ylabel('Loss', fontfamily = 'serif', fontsize = '13' )
    ax1.plot(history['accuracy'], label = 'train',linewidth=2)
    ax1.plot(history['val_accuracy'], label = 'validation',linewidth=2)
    ax2.plot(history['loss'], label = 'train',linewidth=2)
    ax2.plot(history['val_loss'], label = 'validation',linewidth=2)
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.legend(['train', 'validation'], loc='upper left')
    if save:
        if not os.path.isdir(os.path.dirname(output_path_name)):
            os.makedirs(os.path.dirname(output_path_name))
        plt.savefig(output_path_name)
    return fig1

def learning_curve_logger(csv_logger, save= False, output_path_name= None):
    history = pd.read_csv(csv_logger).iloc[:,1:].to_dict('list')
    fig = learning_curve(history, save, output_path_name)
    return fig

def evalution_metrics(test_label, labels_score, save=False, txt_name=None, path = './'):
    accuracy = accuracy_score(test_label, labels_score.round())
    confusion = confusion_matrix(test_label, labels_score.round())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    precision = TP / float(TP + FP)
    sensitivity = TP / float(FN + TP)
    specificity = TN / float(TN + FP)
    f1 = f1_score(test_label, labels_score.round())
    mcc = matthews_corrcoef(test_label, labels_score.round())
    # precision TP / (TP + FP)
    # recall: TP / (TP + FN)
    # specificity : TN / (TN + FP)
    # f1: 2 TP / (2 TP + FP + FN)
    metrics = np.round([accuracy,precision,sensitivity,specificity,f1,mcc],2)
    columns=['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'mcc']
    metrics_dict = dict(zip(columns,metrics))
    if save:
        df = pd.DataFrame(metrics_dict,index=[0])
        df.to_csv(path+'%s_metrics.csv'%txt_name)
        print('  # Accuracy: %f' % accuracy+'\n')
        print('  # Precision: %f' % precision+'\n')  
        print('  # Sensitivity/Recall: %f' % sensitivity+'\n')
        print('  # Specificity: %f' %specificity+'\n')
        print('  # F1 score: %f' % f1+'\n')
        print('  # Matthews Corrcoef:%f' % mcc+'\n')
    else:
        return(metrics_dict)