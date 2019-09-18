from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import argparse

import os
import sys

from mimic3benchmark.subject import read_stays, read_diagnoses, read_events, read_events_tables, get_events_for_stay, add_hours_elpased_to_events, include_hours_elapsed_to_events
from mimic3benchmark.subject import convert_events_to_timeseries, sort_events, get_first_valid_from_timeseries
from mimic3benchmark.preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, read_variable_ranges, clean_events
from mimic3benchmark.preprocessing import transform_gender, transform_ethnicity, assemble_episodic_data


parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
parser.add_argument('--variable_map_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/itemid_to_variable_map.csv'),
                    help='CSV containing ITEMID-to-VARIABLE map.')
parser.add_argument('--mimic', type=str, default='~/MIMIC-III')
parser.add_argument('--reference_range_file', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/variable_ranges.csv'),
                    help='CSV containing reference ranges for VARIABLEs.')
parser.add_argument('--verbose', '-v', type=int, help='Level of verbosity in output.', default=1)
args, _ = parser.parse_known_args()

d_items = pd.read_csv(args.mimic + '/D_ITEMS.csv')
d_tables = {
    'chartevents': d_items,
    'procedureevents_mv': d_items,
    'outputevents': d_items,
    'inputevents_mv': d_items,
    'inputevents_mv': d_items,
    'inputevents_cv': d_items,
    'datetimeevents': d_items,
    'labevents': pd.read_csv(args.mimic + '/D_LABITEMS.csv'),
    'diagnoses_icd': pd.read_csv(args.mimic + '/D_ICD_DIAGNOSES.csv'),
    'procedures_icd': pd.read_csv(args.mimic + '/D_ICD_PROCEDURES.csv')
}

var_map = read_itemid_to_variable_map(args.variable_map_file)
variables = var_map.VARIABLE.unique()

for subject_dir in os.listdir(args.subjects_root_path):
    dn = os.path.join(args.subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        continue
    sys.stdout.write('Subject {}: '.format(subject_id))
    sys.stdout.flush()

    try:
        sys.stdout.write('reading...')
        sys.stdout.flush()
        stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
        diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
        all_events_tables = read_events_tables(os.path.join(args.subjects_root_path, subject_dir))
    except Exception as e:
        sys.stdout.write(f'error reading from disk!: {e}\n')
        continue
    else:
        sys.stdout.write('got {0} stays, {1} diagnoses...'.format(stays.shape[0], diagnoses.shape[0]))
        sys.stdout.flush()

    episodic_data = assemble_episodic_data(stays, diagnoses)

    sys.stdout.write('cleaning and converting to time series...')
    sys.stdout.flush()

    sys.stdout.write(f'extracting separate episodes... in {subject_dir}')
    sys.stdout.flush()

    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        sys.stdout.write(' {}'.format(stay_id))
        sys.stdout.flush()
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        for table, all_events in all_events_tables:
            all_events = sort_events(all_events, variables=variables)
            all_events = get_events_for_stay(all_events, stay_id, intime, outtime)

            if table in d_tables:
                id_columns = [col for col in all_events.columns if col in {'ITEMID', 'ICD9_CODE'}]
                d_table = d_tables[table]
                label_columns = [col for col in d_table.columns if col in {'SHORT_TITLE','LONG_TITLE','LABEL','CATEGORY'}]
                all_events = pd.merge(all_events, d_table[id_columns + label_columns], on=id_columns, how='inner')
            all_events = include_hours_elapsed_to_events(all_events, intime).set_index('HOURS').sort_index(axis=0)

            columns = list(all_events.columns)
            columns_sorted = sorted(all_events, key=(lambda x: "" if x == "Hours" else x))
            all_events = all_events[columns_sorted]

            all_events.to_csv(os.path.join(args.subjects_root_path, subject_dir, 'episode{}_timeseries_{}.csv'.format(i+1, table)), index_label='Hours')

    sys.stdout.write(' DONE!\n')
