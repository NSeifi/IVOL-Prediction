import pandas  as pd
import numpy as np
from pandas.tseries.offsets import *
from tqdm import tqdm
########################################################################################################
# Daily CRSP and Fama-French Merge
########################################################################################################
all_usecols=['PERMNO','date','NAMEENDT','SHRCD','EXCHCD','SICCD','NCUSIP',
                            'TICKER','COMNAM','SHRCLS','TSYMBOL','NAICS','PRIMEXCH','TRDSTAT',
                            'SECSTAT','PERMCO','ISSUNO','HEXCD','HSICCD','CUSIP','DCLRDT','DLAMT',
                            'DLPDT','DLSTCD','NEXTDT','PAYDT','RCRDDT','SHRFLG','HSICMG','HSICIG',
                            'DISTCD','DIVAMT','FACPR','FACSHR','ACPERM','ACCOMP','NWPERM','DLRETX',
                            'DLPRC','DLRET','TRTSCD','NMSIND','MMCNT','NSDINX','BIDLO','ASKHI','PRC',
                            'VOL','RET','BID','ASK','SHROUT','CFACPR','CFACSHR','OPENPRC','NUMTRD',
                            'RETX','vwretd','vwretx','ewretd','ewretx','sprtrn']
print("loading crsp data, this may take a while ...")
crsp_d = pd.read_csv('CRSP-DAILY STOCK COMPLETE.csv', 
                   usecols=['PERMNO','date','SHRCD','NCUSIP', 'TICKER','PERMCO',
                             'CUSIP','BIDLO','ASKHI','PRC', 'VOL','RET','BID','ASK',
                             'RETX','vwretd','vwretx','ewretd','ewretx'])
print("converting date data format ...")
crsp_d['date'] =  pd.to_datetime(crsp_d['date'], format='%d%b%Y')
print("reading fama-french data ...")
ffrench_d = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', usecols=['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
ffrench_d['date'] =  pd.to_datetime(ffrench_d['date'], format='%Y%m%d')
print("merging data ...")
crsp_ffrench_d=pd.merge(crsp_d, ffrench_d, how='inner', on=['date'])
crsp_ffrench_d['year_month']  = crsp_ffrench_d['date'].dt.to_period('M')
grouped_df = crsp_ffrench_d.groupby(['PERMNO', 'CUSIP', 'year_month'])
print("creating pickle files ...")
for key, item in tqdm(grouped_df):
    grouped_df.get_group(key).reset_index().to_pickle("CRSP_FFRENCH_DAILY_GROUPED/{}.pkl".format(str(key[0])+"_"+key[1]+"_"+key[2].strftime("%Y-%m")))
########################################################################################################
# crsp_ffrench_d.to_csv('crsp_ffrench_d.csv')
# crsp_ffrench_d = pd.read_csv('crsp_ffrench_d.csv')
# crsp_ffrench_d['date']  = pd.to_datetime(crsp_ffrench_d['date'], format='%Y-%m-%d')
# crsp_ffrench_d['year_month']  = crsp_ffrench_d['date'].dt.to_period('M')
# grouped_df = crsp_ffrench_d.groupby(['PERMNO', 'CUSIP', 'year_month'])
########################################################################################################
# crsp_ffrench_d[(crsp_ffrench_d.PERMNO==10001) & (crsp_ffrench_d.CUSIP=="36720410") & (crsp_ffrench_d.year_month=="2007-01")]
# FFC = crsp_ffrench_d[(crsp_ffrench_d.PERMNO==10051) & (crsp_ffrench_d.CUSIP=="41043F20") & (crsp_ffrench_d.year_month=="2007-01")]
########################################################################################################
