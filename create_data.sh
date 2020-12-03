    python -m mimic3benchmark.scripts.validate_events data/root/
    python -m mimic3benchmark.scripts.validate_all_events data/root/
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
    python -m mimic3benchmark.scripts.extract_all_episodes_from_subjects data/root/
    python -m mimic3benchmark.scripts.split_train_and_test data/root/
    python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
