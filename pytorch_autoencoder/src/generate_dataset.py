# default package
import os
import numpy as np

# third party package
import wfdb
 
# global parameter
# download dataset
dataset_root = './_data'
download_dir = os.path.join(dataset_root, 'download')
wfdb.dl_database('mitdb', dl_dir=download_dir)
            
# setting
window_size=720  # 2 seconds
sample_rate = 360  # 360 Hz
 
# list
train_record_list = [
        '101', '106', '108', '109', '112', '115', '116', '118', '119', '122',
        '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
        ]
test_record_list = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '210',
        '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
        ]
 
# annotation
labels = ['N', 'V']
valid_symbols = ['N', 'L', 'R', 'e', 'j', 'V', 'E']
label_map = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N','V': 'V', 'E': 'V'}
 

def _load_data(base_record, channel=0):
    record_name = os.path.join(download_dir, str(base_record))
    # read dat file
    signals, fields = wfdb.rdsamp(record_name)
    assert fields['fs'] == sample_rate
    # read annotation file
    annotation = wfdb.rdann(record_name, 'atr')
    symbols = annotation.symbol
    positions = annotation.sample
    return signals[:, channel], symbols, positions
 

def _segment_data(signal, symbols, positions):
    X, y = [], []
    sig_len = len(signal)
    for i in range(len(symbols)):
        start = positions[i] - window_size // 2
        end = positions[i] + window_size // 2
        if symbols[i] in valid_symbols and start >= 0 and end <= sig_len:
            segment = signal[start:end]
            assert len(segment) == window_size, "Invalid length"
            X.append(segment)
            y.append(labels.index(label_map[symbols[i]]))
    return np.array(X), np.array(y)
 

def preprocess_dataset(record_list, mode): 
    Xs, ys = [], []
    save_dir = os.path.join(dataset_root) 
    for i in range(len(record_list)):
        signal, symbols, positions = _load_data(record_list[i])
        signal = (signal - np.mean(signal)) / np.std(signal)
        X, y = _segment_data(signal, symbols, positions)
        Xs.append(X)
        ys.append(y)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "x_"+str(mode)+".npy"), np.vstack(Xs))
    np.save(os.path.join(save_dir, "y_"+str(mode)+".npy"), np.concatenate(ys))


def generate_normal_anomaly_data():
    for mode in ["train","test"]:
        x =np.load(f'{dataset_root}/x_{mode}.npy')
        y =np.load(f'{dataset_root}/y_{mode}.npy')

        for i,state in enumerate(["normal","anomaly"]):
            idx = np.where(y == i)
            np.save(f"{dataset_root}/x_{mode}_{state}.npy",x[idx])
        

if __name__=="__main__":
    preprocess_dataset(train_record_list, "train") 
    preprocess_dataset(test_record_list, "test") 

    generate_normal_anomaly_data()

