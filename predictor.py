import argparse
from tensorflow.keras.models import load_model
from model_training.PC6_encoding import PC_6
from model_training.doc2vec import Doc2Vec_encoding
import joblib
import numpy as np
import pandas as pd

def main(fasta_path, output_csv_name):
    # encoding
    dat = PC_6(fasta_path)
    data_PC6 = np.array(list(dat.values()))
    data_Doc2Vec = Doc2Vec_encoding(fasta_path ,model_path='Doc2Vec_model/AFP_doc2vec.model')

    # reshape
    data_PC6_flatten = data_PC6.reshape(data_PC6.shape[0],-1)

    # load 6 models & predict
    PC6_nn_model = load_model('PC6_model/pc6_final_weights.h5')
    PC6_nn_labels_score = PC6_nn_model.predict(data_PC6).reshape(-1)

    PC6_svc_model = joblib.load('PC6_model/svm_pc6.pkl')
    PC6_svc_labels_score = PC6_svc_model.predict(data_PC6_flatten)

    PC6_rf_model = joblib.load('PC6_model/forest_pc6.pkl')
    PC6_rf_labels_score = PC6_rf_model.predict(data_PC6_flatten)

    Doc2vec_nn_model = load_model('Doc2Vec_model/doc2vec_best_weights.h5')
    Doc2vec_nn_labels_score = Doc2vec_nn_model.predict(data_Doc2Vec).reshape(-1)

    Doc2vec_svc = joblib.load('Doc2Vec_model/svm_doc2vec.pkl')
    Doc2vec_svc_labels_score = Doc2vec_svc.predict(data_Doc2Vec)

    Doc2vec_rf_model = joblib.load('Doc2Vec_model/forest_doc2vec.pkl')
    Doc2vec_rf_labels_score = Doc2vec_rf_model.predict(data_Doc2Vec)

    #ensemble
    avg_ensemble_scores_all = (PC6_nn_labels_score+PC6_svc_labels_score+PC6_rf_labels_score+Doc2vec_nn_labels_score+Doc2vec_svc_labels_score+Doc2vec_rf_labels_score)/6
    # predict
    classifier = avg_ensemble_scores_all>0.5

    # make dataframe
    df = pd.DataFrame(avg_ensemble_scores_all)
    df.insert(0,'Peptide' ,dat.keys())
    df.insert(2,'Prediction results', classifier)
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})
    df = df.rename({0:'Score'}, axis=1)
    # output csv
    df.to_csv(output_csv_name)

#arg
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='AI4AFP ensemble predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta path',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    args = parser.parse_args()
    fasta_path = args.fasta_name
    output_csv_name =  args.output_csv
    main(fasta_path, output_csv_name)
