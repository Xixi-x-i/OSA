import pickle
import sys
import biosppy.signals.tools as st
import numpy as np
import os
import wfdb
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.signal import medfilt
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.signal import cheby2, filtfilt
base_dir = "dataset"

fs = 100
sample = fs * 60  

before = 2  
after = 2  
hr_min = 20
hr_max = 300
def find_p_peaks(ecg_signal, r_peaks, search_window=20, exclude_window=5):

    p_peaks = []

    for r_peak in r_peaks:
        search_start = max(0, r_peak - search_window)
        search_end = max(0, r_peak - exclude_window)
        p_peak_relative = np.argmax(ecg_signal[search_start:search_end])
        p_peak_absolute = p_peak_relative + search_start

        p_peaks.append(p_peak_absolute)

    return p_peaks
def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)
def min_max_normalize(lst):
    minimum = min(lst)
    maximum = max(lst)
    normalized_lst = [(x - minimum) / (maximum - minimum) for x in lst]
    return normalized_lst
def worker(name, labels):
    X = []
    y = []
    groups = []
    signals = wfdb.rdrecord(os.path.join(base_dir, name), channels=[0]).p_signal[:, 0]
    for j in tqdm(range(len(labels)), desc=name, file=sys.stdout):
        if j < before or (j + 1 + after) > len(signals) / float(sample):
            continue
        signal = signals[int((j - before) * sample):int((j + 1 + after) * sample)]
        signal, _, _ = st.filter_signal(signal, ftype='FIR', band='bandpass', order=int(0.3 * fs),
                                        frequency=[3, 45], sampling_rate=fs)
        rpeaks, = hamilton_segmenter(signal, sampling_rate=fs)
        rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
        mask = (rpeaks <= 29950) & (rpeaks >=50)
        rpeaks = rpeaks[mask]
        if len(rpeaks) / (1 + after + before) < 40 or len(rpeaks) / (1 + after + before) > 200:
            continue
        ppeaks = find_p_peaks(signal, rpeaks,20,5)
        new_signal = []
        for value in ppeaks:
            new_signal.append(signal[int(value): int(value) + 28])

        min_distance_list = []
        max_distance_list = []
        all_distances_list = []

        for element_1 in range(len(new_signal)):
            base_array = new_signal[element_1]
            min_distance = np.inf
            max_distance = -np.inf
            distances = []

            for element_2 in range(len(new_signal)):
                if element_1 != element_2:
                    distance = euclidean_distance(base_array, new_signal[element_2])
                    distances.append(distance)

                    if distance < min_distance:
                        min_distance = distance
                    if distance > max_distance:
                        max_distance = distance

            min_distance_list.append(min_distance)
            max_distance_list.append(max_distance)
            all_distances_list.append(distances)

        min_distance_list = min_max_normalize(min_distance_list)
        max_distance_list = min_max_normalize(max_distance_list)
        
        hr_coefficient = np.diff(rpeaks) / float(fs)
        hr_coefficient = medfilt(hr_coefficient, kernel_size=3)
        hr = 60 / hr_coefficient
        min_distance_list = np.array(min_distance_list)
        max_distance_list = np.array(max_distance_list)
        if np.all(np.logical_and(hr >= hr_min, hr <= hr_max)):
            X.append([min_distance_list,max_distance_list])
            y.append(0. if labels[j] == 'N' else 1.)
            groups.append(name)
    return X, y, groups
if __name__ == "__main__":
    apnea_ecg = {}

    names = [
        "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
        "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
        "b01", "b02", "b03", "b04", "b05",
        "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
    ]

    o_train = []
    y_train = []
    groups_train = []
    print('Training...')
    for i in range(len(names)):
        labels = wfdb.rdann(os.path.join(base_dir, names[i]), extension="apn").symbol
        X, y, groups = worker(names[i], labels)
        o_train.extend(X)
        y_train.extend(y)
        groups_train.extend(groups)

    print()

    answers = {}
    with open(os.path.join(base_dir, "event-2-answers.txt"), "r") as f:
        for answer in f.read().split("\n\n"):
            answers[answer[:3]] = list("".join(answer.split()[2::2]))

    names = [
        "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
        "x31", "x32", "x33", "x34", "x35"
    ]

    o_test = []
    y_test = []
    groups_test = []
    print("Testing...")
    for i in range(len(names)):
        labels = answers[names[i]]
        X, y, groups = worker(names[i], labels)
        o_test.extend(X)
        y_test.extend(y)
        groups_test.extend(groups)

    apnea_ecg = dict(
        o_train=o_train, y_train=y_train, groups_train=groups_train,
        o_test=o_test, y_test=y_test, groups_test=groups_test
    )
    with open(os.path.join(base_dir, "apnea.pkl"), "wb") as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print("\nok!")
