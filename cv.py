import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler, RobustScaler, MaxAbsScaler
from scipy.stats import pearsonr
from typing import Any, Tuple
import json
import argparse
from utils import setup_logger, get_processed_dataset, save_preduction_results
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def parse_args():
    parser = argparse.ArgumentParser('Cross-validation')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--task', type=str, default='sentence', help='Task type complexity prediction')
    parser.add_argument('--params', type=str, default= 'ridge_regression.json',
        help='Path to the model parameters file (.json)')
    parser.add_argument('--normalization', type=str, default='l1', help='Normalization type')
    parser.add_argument('--model', type=str,default='lr', help='Model name')
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
        return x_train, x_test
    return x_train, x_test


def get_model(model_name: str, model_params: dict) -> Any:
    if model_name == 'ridge':
        model = Ridge(**model_params)
    elif model_name == 'rf':
        model = RandomForestRegressor(**model_params)
    elif model_name == 'lr':
        model = LinearRegression(**model_params)
    elif model_name == 'svr':
        model = SVR(**model_params)
    elif model_name == 'knn':
        model = KNeighborsRegressor(**model_params)
    elif model_name == 'mlp':
        model = MLPRegressor(**model_params)
    else:
        raise ValueError('Model not supported')
    return model


def get_model_params(params_file_path: str) -> dict:
    os.makedirs('regressor_configurations', exist_ok=True)
    with open(os.path.join("regressor_configurations", params_file_path), 'r') as f:
        model_params = json.load(f)
    return model_params


def cv(x: np.ndarray,
       y: np.ndarray, 
       model_name: str, 
       model_params: dict,
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

    kf = KFold(n_splits=5, shuffle=True, random_state=6)
    scores = []
    mae = []
    pearson = []
    r2 = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = normalize_data(x_train, x_test, normalization)
        model = get_model(model_name, model_params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = y_pred.squeeze()
        y_test = y_test.squeeze()
        scores.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        pearson.append(pearsonr(y_test, y_pred)[0])
        r2.append(r2_score(y_test, y_pred))
    
    return np.mean(scores).round(4).item(), np.mean(mae).round(4).item(), np.mean(pearson).round(4).item(), np.mean(r2).round(4).item()





if __name__ == '__main__':
    args = parse_args()
    data = get_processed_dataset(args.data)

    num_layers = len([k for k  in data[0].keys() if bool(re.search(r'\d', k) and 'token' not in k) ])//2
    results_mean_sentece = {}
    results_last_sentence = {}
    results_nth_token_word = {}
    for layer in range(num_layers):
        if args.task == 'sentence':
            # mean embeddings prediction
            x = np.array([entry[f'embeddgins_{layer}_mean'] for entry in data])
            y = np.array([entry['targets'] for entry in data])
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            mse, mae, pearson, r2 = cv(x, y, args.model, get_model_params(args.params), args.normalization)
            print(f'Layer {layer} embeddings mean: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            results_mean_sentece[layer] = {'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2}
            # last token embeddings prediction
            x = np.array([entry[f'embedding_{layer}_last'] for entry in data])
            mse, mae, pearson, r2 = cv(x, y, args.model, get_model_params(args.params), args.normalization)
            print(f'Layer {layer} embeddings last: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            results_last_sentence[layer] = {'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2}
      
        elif args.task == 'word':
            # nth tokens embeddings prediction
            x = np.array([entry[f'tokens_{layer}'][entry['nth_tokens'][0]: entry['nth_tokens'][1]].mean(axis=0) for entry in data])
            y = np.array([entry['targets'] for entry in data])
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            mse, mae, pearson, r2 = cv(x, y, args.model, get_model_params(args.params), args.normalization)
            print(f'Layer {layer} embeddings mean: MSE: {mse}, MAE: {mae}, Pearson: {pearson}, R2: {r2}')
            results_nth_token_word[layer] = {'mse': mse, 'mae': mae, 'pearson': pearson, 'r2': r2}

    if args.task == "sentence":
        save_preduction_results(results=results_mean_sentece,
                            path=f'mean_embeddings_{args.task}_{args.model}_{args.normalization}_{".".join("_".join(args.data.split("/")).split(".")[:-1])}.json')
        save_preduction_results(results=results_last_sentence,
                                path=f'last_embeddings_{args.task}_{args.model}_{args.normalization}_{".".join("_".join(args.data.split("/")).split(".")[:-1])}.json')
    elif args.task == "word":
        save_preduction_results(results=results_nth_token_word,
                                path=f'tokens_embeddings_{args.task}_{args.model}_{args.normalization}_{".".join("_".join(args.data.split("/")).split(".")[:-1])}.json')    
    
