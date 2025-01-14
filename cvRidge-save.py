import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import SVR 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
from typing import Any, Tuple
import json
import argparse
from utils import setup_logger, get_processed_dataset, save_preduction_results
import os
import re
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import yaml


def parse_args():
    parser = argparse.ArgumentParser('Cross-validation')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--task', type=str, default='word', help='Task type complexity prediction')
    parser.add_argument('--normalization', type=str, default='l1', help='Normalization type')
    parser.add_argument('--model', type=str,default='ridge', help='Model name')
    return parser.parse_args()


def normalize_data(x_train: np.ndarray, x_test: np.ndarray, normalization: str = 'l1') -> np.ndarray:
    if normalization == 'l1':
        x_train = normalize(x_train, norm='l1')
        x_test = normalize(x_test, norm='l1')
    elif normalization == 'l2':
        x_train = normalize(x_train, norm='l2')
        x_test = normalize(x_test, norm='l2')
    elif normalization == 'standard':
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif normalization == 'robust':
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif normalization == 'maxabs':
        scaler = MaxAbsScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        raise ValueError('Normalization type not supported')
    return x_train, x_test


def get_model(model_name: str, model_params: dict) -> Any:
    if model_name == 'ridge':
        model = Ridge()
    else:
        raise ValueError('Model not supported')
    return model


def get_model_params(params_file_path: str) -> dict:
    os.makedirs('regressor_configurations', exist_ok=True)
    with open(os.path.join("regressor_configurations", params_file_path), 'r') as f:
        model_params = json.load(f)
    return model_params
    

def save_results_to_file(results: dict, path: str) -> None:
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    results = convert_numpy(results)
    os.makedirs('results', exist_ok=True)
    try:
        with open(os.path.join('results', path), 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"Error saving results to {path}: {e}")



def cv(x: np.ndarray,
       y: np.ndarray, 
       model_name: str, 
       normalization: str = 'l1') -> Tuple[float, float, float, float]:
    """
    Perform cross-validation for a given model and data.
    :param x: np.ndarray, features
    :param y: np.ndarray, targets
    :param model_name: str, name of the model
    :param model_params: dict, model parameters
    :param normalization: str, normalization type
    :return: Tuple[float, float, float, float], mean squared error, mean absolute error, pearson correlation, r2 score
    """
    # grid search
    #alphas= [10**-7,10**-6,10**-5.0,10**-4,10**-3,10**-2.8,10**-2.6,10**-2.4,10**-2.2,10**-2,10**-1.5,10**-1.0,1,10,10**2.0,10**3.0]
    alphas= [10**-3,10**-2.8,10**-2.6,10**-2.4,10**-2.2,10**-2,10**-1.8,10**-1.5,10**-1,10,10**2.0]
    parameters = {'alpha' : alphas}
    model= GridSearchCV(Ridge(), parameters, scoring='neg_mean_squared_error',cv=5)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test = normalize_data(x_train, x_test, normalization)
    model.fit(x_train, y_train)
    #results = model.cv_results_['mean_test_score']
    '''
    plt.plot(np.log10(alphas),-results)
    plt.xlabel("Log10(Alpha)")
    plt.ylabel('test MSE')
    plt.show()
    '''
    y_pred = model.predict(x_test)
    score=mean_squared_error(y_test, y_pred)
    mae=mean_absolute_error(y_test, y_pred)
    pearson=pearsonr(y_test.squeeze(), y_pred.squeeze())[0]
    r2=r2_score(y_test, y_pred)
    print(model.best_params_)
    return score,mae, pearson, r2

if __name__ == '__main__':
    args = parse_args()
    data = get_processed_dataset(args.data)
    avgpearsonMean = 0
    avgpearsonLast = 0
    avgpearson = 0
    num_layers = len([k for k  in data[0].keys() if bool(re.search(r'\d', k) and 'token' not in k) ])//2
    for layer in range(num_layers):
        if args.task == 'sentence':
            # mean embeddings prediction
            x = np.array([entry[f'embeddgins_{layer}_mean'] for entry in data])
            y = np.array([entry['targets'] for entry in data])
            mse, mae, pearson, r2 = cv(x, y, args.model, args.normalization)
            avgpearsonMean+=pearson
            print(f'Layer {layer} embeddings mean: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            save_results_to_file(results={'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2},
                                    path=f'cv_results_mean_embeddings_{args.task}_{args.model}_{args.normalization}_{layer}_\
                                        {".".join("_".join(args.data.split("/")).split(".")[:-1])}.yaml')
            # last token embeddings prediction
            x = np.array([entry[f'embedding_{layer}_last'] for entry in data])
            mse, mae, pearson, r2 = cv(x, y, args.model, args.normalization)
            avgpearsonLast+=pearson
            print(f'Layer {layer} embeddings last: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            save_results_to_file(results={'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2},
                                    path=f'cv_results_last_token_{args.task}_{args.model}_{args.normalization}_{layer}_\
                                        {".".join("_".join(args.data.split("/")).split(".")[:-1])}.yaml')

        elif args.task == 'word':
            # nth tokens embeddings prediction
            x = np.array([entry[f'tokens_{layer}'][max(0,entry['nth_tokens'][0]) : min(entry['nth_tokens'][1],len(entry[f'tokens_{layer}'])) ].mean(axis=0) for entry in data])
            y = np.array([entry['targets'] for entry in data])
            y = y.astype(np.float32)
            mse, mae, pearson, r2 = cv(x, y, args.model, args.normalization)
            avgpearson+=pearson
            print(f'Layer {layer} embeddings mean: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            save_results_to_file(results={'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2},
                                    path=f'cv_results_tokens_embeddings_{args.model}_{args.normalization}_layer_{layer}_\
                                        {".".join("_".join(args.data.split("/")).split(".")[:-1])}.yaml')
    if args.task == 'word': 
        print(avgpearson/num_layers)
        save_results_to_file(results={'avgpearsonWord:' : avgpearson/num_layers},path=f'cv_results_tokens_embeddings_{args.model}_{args.normalization}\
                                        {".".join("_".join(args.data.split("/")).split(".")[:-1])}.yaml')
    else:
        print(avgpearsonMean/num_layers)
        print(avgpearsonLast/num_layers)
        save_results_to_file(results={'avgpearsonMean:' : avgpearsonMean/num_layers,'avgpearsonLast:' : avgpearsonLast/num_layers},path=f'cv_results_tokens_embeddings_{args.model}_{args.normalization}\
                                        {".".join("_".join(args.data.split("/")).split(".")[:-1])}.yaml')
            
    
