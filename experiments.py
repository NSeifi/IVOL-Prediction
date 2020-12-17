"""
Author: Nasrin Seifi
Input:  CRSP_DAIL , FF3 and IVOL dataset
Functionality: IVOL Prediction
Output: Trained Model

Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        preprocess/ivol_split.py
"""

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from predictor import Predictor, device

# #######################################################################################################
# ##### Experiment #1
# #######################################################################################################
learning_momentum = 0.99
lr = 0.005
max_grad_norm = 1.0
# batch size
cumulate_loss = 16
experiment_test_start_year = 2019
experiment_test_end_year = 2019
experiment_train_start_year = 2018
experiment_train_end_year = 2018
# Input columns from Merged dataset
experiment_features = ['Mkt-RF', 'SMB', 'HML', 'RF', 'vwretx', 'VOL', 'RET']
target_label = ['IVOL']
# TODO recalculate these once IVOL dataset changed using get_IVOL_boundaries function
min_ivol = 0.0001858190417677868
max_ivol = 8.53472180221995

min_RET = -0.965248
max_RET = 0.9


def read_and_merge(year):
    crsp_d = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year), usecols=[
        'PERMNO', 'date', 'SHRCD', 'NCUSIP', 'TICKER', 'PERMCO', 'CUSIP', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET', 'BID', 'ASK', 'RETX', 'vwretd',
        'vwretx', 'ewretd', 'ewretx'])
    crsp_d['date'] = pd.to_datetime(crsp_d['date'], format='%d%b%Y')
    ffrench_d = pd.read_csv('F-F-DAILY-BY-YEAR/{}.csv'.format(year), usecols=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
    ffrench_d['date'] = pd.to_datetime(ffrench_d['date'], format='%Y%m%d')

    crsp_ffrench_d = pd.merge(crsp_d, ffrench_d, how='inner', on=['date'])
    crsp_ffrench_d['year_month'] = crsp_ffrench_d['date'].dt.to_period('M').astype(str)

    ivol = pd.read_csv('IVOL-BY-YEAR/{}.csv'.format(year), usecols=['PERMNO', 'CUSIP', 'year', 'month', 'year_month', 'IVOL'])
    merged_dataset = pd.merge(crsp_ffrench_d, ivol, how='inner', on=['PERMNO', 'CUSIP', 'year_month'])
    for x in experiment_features:
        merged_dataset[x] = pd.to_numeric(merged_dataset[x], errors='coerce').fillna(0)
    return merged_dataset


def inline_normalize(merged_dataset):
    normalization_params = {}
    for x in experiment_features:
        normalization_params[x] = {'min': merged_dataset[x].min(), 'max': merged_dataset[x].max(), 'mean': merged_dataset[x].mean()}
        merged_dataset[x] = (merged_dataset[x] - normalization_params[x]['min']) / (normalization_params[x]['max'] - normalization_params[x]['min'])
        normalization_params[x]['mean'] = merged_dataset[x].mean()
        # normalization_params[x]['std'] = X[x].std()
        # X[x] = (X[x] - normalization_params[x]['mean']) / normalization_params[x]['std']
        merged_dataset[x] = (merged_dataset[x] - normalization_params[x]['mean'])


def get_IVOL_boundaries(start_year, end_year):
    attribute_name = 'IVOL'
    overall_min = float("inf")
    overall_max = float("-inf")
    for year in range(start_year, end_year + 1):
        ivol = pd.read_csv('IVOL-BY-YEAR/{}.csv'.format(year), usecols=['PERMNO', 'CUSIP', 'year', 'month', 'year_month', 'IVOL'])
        min_ = ivol[attribute_name].min()
        max_ = ivol[attribute_name].max()
        if min_ < overall_min:
            overall_min = min_
        if max_ > overall_max:
            overall_max = max_
    return overall_min, overall_max


def get_RET_boundaries(start_year, end_year):
    attribute_name = 'RET'
    overall_min = float("inf")
    overall_max = float("-inf")
    for year in range(start_year, end_year + 1):
        ret = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year), usecols=['PERMNO', 'CUSIP' , 'RET'])
        ret[attribute_name] = pd.to_numeric(ret[attribute_name], errors='coerce').fillna(0)
        min_ = ret[attribute_name].astype(float).min()
        max_ = ret[attribute_name].astype(float).max()
        if min_ < overall_min:
            overall_min = min_
        if max_ > overall_max:
            overall_max = max_
    return overall_min, overall_max

def get_stock_data(start_year, end_year):
    """
    :param start_year: inclusive
    :param end_year: inclusive
    """
    for year in range(start_year, end_year + 1):
        crsp_ffrench_ivol = read_and_merge(year)
        inline_normalize(crsp_ffrench_ivol)
        # CRSP_FF3_IVOL for each stock in each month
        crsp_ffrench_ivol_grouped_df = crsp_ffrench_ivol.groupby(['PERMNO', 'CUSIP', 'year_month'])
        # key= (PERMNO, CUSIP, YEAR_MONTH)
        for key, item in crsp_ffrench_ivol_grouped_df:
            X_all = crsp_ffrench_ivol_grouped_df.get_group(key)
            # X has the feature labels
            X = X_all[experiment_features].astype(float)
            X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
            # Y is a tensor of target values (vector)
            Y = torch.FloatTensor(X_all[target_label].astype(float).iloc[0].to_numpy()).to(device)
            yield X_tensor, Y


def experiment_1():
    """
     This experiment tries to predict the values of IVOL from the data by directly looking at the raw variables.
    """
    # Instance from Predictor Class
    p = Predictor(len(experiment_features), 32, len(target_label), 1).to(device)
    # Stochastic Gradient Descent optimizer
    optimizer = torch.optim.SGD(p.parameters(), lr=lr, momentum=learning_momentum)
    # Temporary variables for computing the average loss
    all_loss = 0.0
    all_count = 0
    # range from 1968 to 2019
    dt = tqdm(get_stock_data(experiment_train_start_year, experiment_train_end_year))
    optimizer.zero_grad()
    for X, Y in dt:
        # Prediction
        _, loss = p(X, Y)
        all_loss += loss.item()
        all_count += 1
        loss.backward()
        if all_count and all_count % cumulate_loss == 0:
            nn.utils.clip_grad_norm_(p.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        dt.set_description("Average Loss: {:.2f}".format(all_loss / all_count))


def categorize(Y):
    value = Y.item()
    ivol_step = (max_ivol - min_ivol / 3.0)
    if value < min_ivol + ivol_step:
        return torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 2 * ivol_step:
        return torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 3 * ivol_step:
        return torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 4 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 5 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 6 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).to(device)
    elif value < min_ivol + 7 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).to(device)
    elif value < min_ivol + 8 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).to(device)
    elif value < min_ivol + 9 * ivol_step:
        return torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).to(device)
    else:
        return torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).to(device)


def convert_class_back_to_label(class_id):
    if class_id == 0:
        return "low"
    elif class_id < 5:
        return "medium"
    else:
        return "high"


def experiment_2():
    """
    This experiment divides the range of calculated IVOLs to three categories of low, medium, and high
     and tries to predict those categories for unseen data.
    """
    # Instance from Predictor Class
    p = Predictor(len(experiment_features), 32, len(target_label), 10).to(device)
    # Stochastic Gradient Descent optimizer
    optimizer = torch.optim.SGD(p.parameters(), lr=lr, momentum=learning_momentum)
    # Temporary variables for computing the average loss
    all_loss = 0.0
    all_count = 0
    # range from 1968 to 2019
    print("Training on data from {} to {}".format(experiment_train_start_year, experiment_train_end_year))
    train = tqdm(get_stock_data(experiment_train_start_year, experiment_train_end_year))
    optimizer.zero_grad()
    for X, Y in train:
        y_ct = categorize(Y)
        # Prediction
        _, loss = p(X, y_ct)
        all_loss += loss.item()
        all_count += 1
        loss.backward()
        if all_count and all_count % cumulate_loss == 0:
            nn.utils.clip_grad_norm_(p.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        train.set_description("Average Loss: {:.2f}".format(all_loss / all_count))
    print("Testing on data from {} to {}".format(experiment_test_start_year, experiment_test_end_year))
    test = tqdm(get_stock_data(experiment_test_start_year, experiment_test_end_year))
    res = {"overall": {"correct": 0.0, "total": 0.0},
           "low": {"correct": 0.0, "total": 0.0},
           "medium": {"correct": 0.0, "total": 0.0},
           "high": {"correct": 0.0, "total": 0.0}}
    with torch.no_grad():
        for X, Y in test:
            y_ct = categorize(Y)
            # Prediction
            pred, loss = p(X, y_ct)
            prediction_class = convert_class_back_to_label(pred.argmax().item())
            actual_class = convert_class_back_to_label(y_ct.argmax().item())
            res["overall"]["total"] += 1.0
            res[actual_class]["total"] += 1.0
            if prediction_class == actual_class:
                res["overall"]["correct"] += 1.0
                res[actual_class]["correct"] += 1.0
    for key in res:
        total = res[key]["total"]
        correct = res[key]["correct"]
        if total > 0.0:
            print("{} prediction accuracy: {}% [{}/{}]".format(key, correct * 100 / total, correct, total))


if __name__ == '__main__':
    # print(get_IVOL_boundaries(experiment_train_start_year, experiment_train_end_year))
    # print(get_RET_boundaries(experiment_train_start_year, experiment_train_end_year))
    experiment_2()

