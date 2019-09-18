from __future__ import absolute_import
from __future__ import print_function

import os
import re
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)


def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0,
                      eps=1e-6, future_time_interval=24.0):

    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            if "_all.csv" in ts_filename:
                continue

            lb_filename = re.sub(r"_timeseries[a-z_]*", "", ts_filename)
            try:
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            except:
                print("Label file EMPTY.")
                continue

            # empty label file
            if label_df.shape[0] == 0:
                break

            mortality = int(label_df.iloc[0]["Mortality"])

            los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
            if pd.isnull(los):
                print("(length of stay is missing)", patient, ts_filename)
                continue

            stay = stays_df[stays_df.ICUSTAY_ID == label_df.iloc[0]['Icustay']]
            deathtime = stay['DEATHTIME'].iloc[0]
            intime = stay['INTIME'].iloc[0]
            if pd.isnull(deathtime):
                lived_time = 1e18
            else:
                lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

            df = pd.read_csv(os.path.join(patient_folder, ts_filename))
            header = df.columns
            ts_lines = df[(-eps < df["Hours"]) & (df["Hours"] < los + eps)]
            event_times = ts_lines.Hours

            if len(ts_lines) == 0:
                print("(no events in ICU) ", patient, ts_filename)
                continue

            sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)

            sample_times = list(filter(lambda x: x > shortest_length, sample_times))

            # At least one measurement
            sample_times = list(filter(lambda x: x > event_times.iloc[0], sample_times))

            output_ts_filename = patient + "_" + ts_filename
            ts_lines.to_csv(os.path.join(output_dir, output_ts_filename), index=False)

            for t in sample_times:
                if mortality == 0:
                    cur_mortality = 0
                else:
                    cur_mortality = int(lived_time - t < future_time_interval)
                xty_triples.append((output_ts_filename, t, cur_mortality))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')

    print(len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},{:.6f},{:d}\n'.format(x, t, y))


def main():
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
