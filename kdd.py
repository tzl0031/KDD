import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from time import time
# from matplotlib import pyplot
import numpy as np

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

FTEST = "/home/tian/Github/KDD/kddcup.data_10_percent_corrected"
FTRAIN = "/home/tian/Github/KDD/kddcup.data_10_percent_corrected"


dos = ["back.", "land.", "neptune.", "pod.", "smurf.", "teardrop."]
r2l = ["ftp_write.", "guess_passwd.", "imap.", "multihop." ,"phf.", "spy.", "warezclient.", "warezmaster."]
u2r = ["buffer_overflow.", "loadmodule.", "perl.", "rootkit."]
probe = ["ipsweep.", "nmap.", "portsweep.", "satan."]

class DataSet(object):
    def __init__(self, data, label):
        self._num_examples = data.shape[0]
        self._data = data
        self._label = label
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._data

    @property
    def labels(self):
        return self._label

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    """
	This method divide the whole sample into batches,
	and it ignores the final residuals.
	"""

    # sample 128 data each time. once reach the end, shuffle the data.
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._label = self._label[perm]
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data[start:end], self._label[start:end]


def load(test=False):
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(fname, header=None, names=col_names)
    data = df[0:42]
    print data.describe()

    if not test:
        label = df["label"].copy()
        label.replace("normal.", "norm", inplace=True)
        label.replace(probe, "probe", inplace=True)
        label.replace(dos, "dos", inplace=True)
        label.replace(u2r, "u2r", inplace=True)
        label.replace(r2l, "r2l", inplace=True)
    else:
        label = None
    return data, label


def feature(data):
    feature_name = [
        "duration", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
    feature = data[feature_name].astype(float)
    feature.apply(lambda x: MinMaxScaler().fit_transform(x))
    return feature


def data_split(valid=0.0):
    class DataSets(object):
        pass

    data_sets = DataSets()
    data, label = load()
    test_data, test_label = load(test=True)
    # Validation ratio
    n = int(valid * len(data))
    valid_data = data[:n]
    valid_label = data[:n]
    train_data = data[n:]
    train_label = data[n:]
    data_sets.valid = DataSet(valid_data, valid_label)
    data_sets.train = DataSet(train_data, train_label)
    data_sets.test = DataSet(test_data, test_label)
    return data_sets



# def data_scaling(data, label):

if __name__ == '__main__':
    data, label = load()
    feature  = feature(data)
    print feature.describe()

    # feature(data)
