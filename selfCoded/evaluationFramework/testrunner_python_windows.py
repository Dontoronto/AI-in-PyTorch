import os
import subprocess
import sys

# Define the configurations
BASE_SAVE_PATH = os.path.join("experiment", "adv_data", "ResNet18")
ATTACKS_CONFIG = os.path.join("configs", "adversarial_configs", "AttacksConfig.json")
PROVIDER_CONFIG = os.path.join("configs", "adversarial_configs", "ProviderConfig.json")
THREAT_MODEL = os.path.join("configs", "adversarial_configs", "ThreatModelConfig.json")
TRAINER_CONFIG = os.path.join("configs", "TrainerConfig.json")
DATAHANDLER_CONFIG = os.path.join("configs", "DataHandlerConfig.json")
ARCHITECTURE_CONFIG = os.path.join("configs", "preOptimizingTuning", "ADMModelArchitecture.json")
ADMM_CONFIG = os.path.join("configs", "preOptimizingTuning", "ADMMConfig.json")
ANALYZER_CONFIG = os.path.join("configs", "analyzer", "AnalyzerConfig.json")
PYTHON_DEFAULT = "python testrunner_change_config.py"
PYTHON_ARCHITECTURE = "python testrunner_change_architecture_config.py"
PYTHON_ATTACKS = "python testrunner_change_attacks_config.py"
MAIN_COMMAND = "python main.py"

# List of changes after each test run

CHANGES = [
    # =========== Note: PGD
    #f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true " +
    #f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=50000 &&"
    #f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} PGD:eps=5.0 PGD:alpha=0.00205",

    # =========== Note: PGD l2_norm 10 u. 100 thesis example
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=14004 adv_settings.adv_sample_range_end=14005 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} PGD:eps=0.0 PGD:alpha=0.00",
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=14004 adv_settings.adv_sample_range_end=14005 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} PGD:eps=50.0 PGD:alpha=0.090",
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=14004 adv_settings.adv_sample_range_end=14005 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} PGD:eps=50.0 PGD:alpha=0.0205",
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=14004 adv_settings.adv_sample_range_end=14005 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} PGD:eps=5.0 PGD:alpha=0.00205",



    # ============ Note: JSMA
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=2 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=2 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} JSMA:theta=0.5 JSMA:gamma=0.1",
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} JSMA:theta=1.0 JSMA:gamma=0.06",
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} JSMA:theta=1.0 JSMA:gamma=0.1",
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} JSMA:theta=1.0 JSMA:gamma=0.2"


    # ============= Note: DeepFool
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=0 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=50000 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} DeepFool:steps=1 DeepFool:overshoot=14.0",

    # # =========== Note: OnePixel
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=3 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=50000 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} OnePixel:pixels=1 OnePixel:steps=10 OnePixel:popsize=10",


    # =========== Note: SparseFool
    # f"{PYTHON_DEFAULT} {ANALYZER_CONFIG} adv_settings.adv_attack_selection=4 adv_settings.adv_save_enabled=true " +
    # f"adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=100 &&"
    # f"{PYTHON_ATTACKS} {ATTACKS_CONFIG} SparseFool:steps=1 SparseFool:lam=3 OnePixel:overshoot=0.02",

]
# Number of test runs
NUM_RUNS = len(CHANGES)

for i in range(NUM_RUNS):
    run_save_path = os.path.join(BASE_SAVE_PATH, f"run_{i}")
    os.makedirs(run_save_path, exist_ok=True)

    # Update the configuration file for the current test run
    print(f"Updating configuration file for test run {i}...")
    command = f"{CHANGES[i]} && {PYTHON_DEFAULT} {ANALYZER_CONFIG} save_path={run_save_path}"
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error updating the configuration file after test run {i}!")
        sys.exit(1)

    print(f"Starting test run {i}...")

    # Execute the test run
    result = subprocess.run(MAIN_COMMAND, shell=True)

    if result.returncode != 0:
        print(f"Test run {i} failed!")
        sys.exit(1)

    print(f"Test run {i} completed.")
    print(f"Configuration file successfully updated after test run {i}.")

print("All test runs successfully completed.")
