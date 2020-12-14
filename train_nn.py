'''
Author: Nasrin Seifi
Input:  CRSP_DAIL , FF3 and IVOL dataset
Functionality: IVOL Prediction
Output: Trained Model

Execution Comment:
    - before running this code, run the following scripts to prepare the required pre-processed data for this script:
        preprocess/ivol_split.py
'''

import pandas  as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #######################################################################################################
# ##### Experiment #1
# #######################################################################################################
learning_momentum = 0.99
lr = 0.005
max_grad_norm = 1.0

class Predictor(nn.Module):
    def __init__(self, n_experiment_features = 7, n_hidden_layer = 32, n_out_labels = 1, encoder_layers = 2):
        super(Predictor, self).__init__()
        self.emb = nn.Linear(n_experiment_features, n_hidden_layer, bias=True)
        self.encoder = nn.LSTM(n_hidden_layer, n_hidden_layer, encoder_layers)
        self.generator = nn.Linear(n_hidden_layer, n_out_labels, bias=True)
        self.encoder_layers = encoder_layers
        self.encoder_bidirectional = False
        self.encoder_hidden = n_hidden_layer
        self.criterion = nn.MSELoss()

    def forward(self, input_tensor, target_tensor=None):
        e = self.emb(input_tensor).unsqueeze(1)
        init_context = self.encoder_init(1)
        encoded, output_context = self.encoder(e, init_context)
        pred = self.generator(output_context[0]).view(-1)
        loss = 0
        if target_tensor is not None:
            loss = self.criterion(pred, target_tensor)
        return pred, loss

    def encoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32)

# Input columns from Merged dataset
experiment_features = ['Mkt-RF', 'SMB', 'HML', 'RF', 'vwretx', 'VOL', 'RET']
target_label = ['IVOL']
# Instance from Predictor Class
p = Predictor(len(experiment_features), 32, len(target_label), 1).to(device)
# Stochastic Gradient Descent optimizer
optimizer = torch.optim.SGD(p.parameters(), lr=lr, momentum=learning_momentum)
# Temporary variables for computing the average loss
all_loss = 0.0
all_count = 0
# batch size
cumulate_loss = 16
# range from 1968 to 2019
for year in range(2019,2020):
    crsp_d = pd.read_csv('CRSP-DAILY-BY-YEAR/{}.csv'.format(year),
                         usecols=['PERMNO','date','SHRCD','NCUSIP', 'TICKER','PERMCO',
                                  'CUSIP','BIDLO','ASKHI','PRC', 'VOL','RET','BID','ASK',
                                  'RETX','vwretd','vwretx','ewretd','ewretx'])
    crsp_d['date'] =  pd.to_datetime(crsp_d['date'], format='%d%b%Y')
    ffrench_d = pd.read_csv('F-F-DAILY-BY-YEAR/{}.csv'.format(year), usecols=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
    ffrench_d['date'] =  pd.to_datetime(ffrench_d['date'], format='%Y%m%d')

    crsp_ffrench_d=pd.merge(crsp_d, ffrench_d, how='inner', on=['date'])
    crsp_ffrench_d['year_month']  = crsp_ffrench_d['date'].dt.to_period('M').astype(str)

    ivol = pd.read_csv('IVOL-BY-YEAR/{}.csv'.format(year), usecols=['PERMNO', 'CUSIP', 'year', 'month', 'year_month', 'IVOL'])
    crsp_ffrench_ivol = pd.merge(crsp_ffrench_d, ivol, how='inner', on=['PERMNO', 'CUSIP', 'year_month'])

    X = crsp_ffrench_ivol # []
    X['RET'] = pd.to_numeric(X['RET'], errors='coerce').fillna(0)
    ###################################
    # Normalization step
    ###################################
    normalization_params = {}
    for x in experiment_features:
        normalization_params[x] = {'min': X[x].min(), 'max': X[x].max(), 'mean': X[x].mean()}
        X[x] = (X[x] - normalization_params[x]['min']) / (normalization_params[x]['max'] - normalization_params[x]['min'])
        normalization_params[x]['mean'] = X[x].mean()
        # normalization_params[x]['std'] = X[x].std()
        # X[x] = (X[x] - normalization_params[x]['mean']) / normalization_params[x]['std']
        X[x] = (X[x] - normalization_params[x]['mean'])
    # CRSP_FF3_IVOL for each stock in each month
    crsp_ffrench_ivol_grouped_df = crsp_ffrench_ivol.groupby(['PERMNO', 'CUSIP', 'year_month'])
    dt = tqdm(crsp_ffrench_ivol_grouped_df)
    optimizer.zero_grad()

    # Key= (PERMNO, CUSIP, YEAR_MONTH)
    for key, item in dt:
        X_all = crsp_ffrench_ivol_grouped_df.get_group(key)
        # X has the feature labels
        X = X_all[experiment_features].astype(float)
        X_tensor = torch.FloatTensor(X.to_numpy()).to(device)
        # Y is a tensor of target values (vector)
        Y = torch.FloatTensor(X_all[target_label].astype(float).iloc[0].to_numpy()).to(device)
        # Prediction
        _, loss = p(X_tensor, Y)
        all_loss += loss.item()
        all_count +=1
        loss.backward()
        if all_count and all_count % cumulate_loss == 0:
            nn.utils.clip_grad_norm_(p.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        dt.set_description("Average Loss: {:.2f}".format(all_loss / all_count))
