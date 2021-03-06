# %%
import pandas as pd
import os
import numpy as np
import pickle
import pandas as pd
import logging

logger = logging.getLogger()

path = './data/mock/raw/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]

print(file_list_py)
# %%

for file in file_list_py :
    vm1 = pd.read_csv(path + file, usecols=['mean_cpu_usage'])
    if vm1.shape[0] < 12780:
        continue
    vm2 = vm1.copy()
    vm2['mean_cpu_usage'] = 0 
    vm1 = vm1.reset_index()
    # vm1 = vm1.append(vm1).append(vm1).reset_index()  3배로 늘림
    # anomaly = [0.9 * np.sin(x * np.pi/240) for x in range(240)]

    # vm1['is_anomaly'] = 0

    # for i in range(240):
    #     vm1['mean_cpu_usage'][1320+i::12*240] += anomaly[i]
        # vm1['is_anomaly'][132+i::12*24] = 1

    vm1[vm1['mean_cpu_usage']>1] = 1
    # print(len(vm1[vm1['mean_cpu_usage']>0.8]))

    vm1['mean_mem_usage'] = vm1.mean_cpu_usage
    vm1['mean_disk_usage'] = vm1.mean_cpu_usage
    vm1['mean_network_usage'] = vm1.mean_cpu_usage
    vm1.drop(columns=['index'],inplace=True)
    vm1 = vm1.iloc[:12000, :]
    vm1 = vm1.fillna(method='ffill')

    # save file 
    hostname = file.rstrip('.csv')
    vm1.to_pickle(f'./data/mock/processed/{hostname}.pkl')


# %%
# Data check
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

number = randint(0,65)
data_path = './data/mock/processed/'
file_list = [host for host in os.listdir(data_path) if host.endswith('pkl')]
print(file_list[number])
host_sample_data = pd.read_pickle(f'{data_path}/{file_list[number]}')


# gca stands for 'get current axis'
# ax = plt.gca()
host_sample_data.iloc[:,:].plot(kind='line',y='mean_cpu_usage')
# host_sample_data.plot(kind='line',x='name',y='mean_cpu_usag', color='red', ax=ax)

plt.show()

    

# %%
#
# REAL DATA PREPROCESSING
#

# Data Check
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

number = randint(0,65)
data_path = './data/vms/'
file_list = [host for host in os.listdir(data_path) if host.endswith('csv')]
# number=2
print(file_list[number])
host_sample_data = pd.read_csv(f'{data_path}/{file_list[number]}')

cols = [
    'system_load_norm_1m',
    'system_load_norm_5m',
    'system_load_norm_15m',
    'system_cpu_total_pct',
    'system_cpu_iowait_pct',
    'system_memory_actual_used_pct',
    'system_memory_used_pct',
    'system_diskio_iostat_write_await',
    'system_diskio_iostat_service_time',
    'system_diskio_iostat_await',
    'system_diskio_iostat_busy',
    'system_diskio_iostat_request_avg_size',
] 

host_sample_data = host_sample_data[cols]
# host_sample_data
# gca stands for 'get current axis'
# ax = plt.gca()

# fig = plt.figure(4,5)
for i in range(len(cols)) :
    plt.plot(host_sample_data[cols[i]])
    plt.title(cols[i])
    plt.show() 
    
# host_sample_data.iloc[:,:].plot(kind='line',y=cols[6])
# host_sample_data.plot(kind='line',x='name',y='mean_cpu_usag', color='red', ax=ax)

 # %%
path = './data/real/raw/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]

for file in file_list_py :
    hostname = file.rstrip('.csv')
    print(hostname)
# %%
cols = [
    'system_load_norm_1m',
    'system_load_norm_5m',
    'system_load_norm_15m',
    'system_cpu_total_pct',
    'system_cpu_iowait_pct',
    'system_memory_actual_used_pct',
    'system_memory_used_pct',
    'system_diskio_iostat_write_await',
    'system_diskio_iostat_service_time',
    'system_diskio_iostat_await',
    'system_diskio_iostat_busy',
    'system_diskio_iostat_request_avg_size',
] 

for file in file_list_py :
    vm1 = pd.read_csv(path + file, usecols=cols)
    vm1 = vm1.reset_index()
    vm1.drop(columns=['index'],inplace=True)    # save file 

    hostname = file.rstrip('.csv')
    vm1.to_pickle(f'./data/real/processed/{hostname}.pkl')


# %%
# Data check
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

number = randint(0,65)
data_path = './data/real/processed/'
file_list = [host for host in os.listdir(data_path) if host.endswith('pkl')]
print(file_list[number])
host_sample_data = pd.read_pickle(f'{data_path}/{file_list[number]}')


# gca stands for 'get current axis'
# ax = plt.gca()
host_sample_data.iloc[:,:].plot(kind='line',y='system_cpu_total_pct')
# host_sample_data.plot(kind='line',x='name',y='mean_cpu_usag', color='red', ax=ax)

plt.show()

