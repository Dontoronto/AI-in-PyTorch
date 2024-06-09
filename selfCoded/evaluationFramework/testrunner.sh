#!/bin/bash

BASE_SAVE_PATH="experiment/adversarial_data/LeNet"
ATTACKS_CONFIG="configs/adversarial_configs/AttacksConfig.json"
PROVIDER_CONFIG="configs/adversarial_configs/ProviderConfig.json"
THREAT_MODEL="configs/adversarial_configs/ThreatModelConfig.json"
TRAINER_CONFIG="configs/TrainerConfig.json"
DATAHANDLER_CONFIG="configs/DataHandlerConfig.json"
ARCHITECTURE_CONFIG="configs/preOptimizingTuning/ADMModelArchitecture.json"
ADMM_CONFIG="configs/preOptimizingTuning/ADMMConfig.json"
ANALYZER_CONFIG="configs/analyzer/AnalyzerConfig.json"
PYTHON_DEFAULT="python testrunner_change_config.py"
PYTHON_ARCHITECTURE="python testrunner_change_architecture_config.py"
PYTHON_ATTACKS="python testrunner_change_attacks_config.py"
MAIN_COMMAND="python main.py"  # Der Befehl zum Ausführen der Tests

# Liste der Änderungen nach jedem Testlauf
#declare -a CHANGES=(
#    "$PYTHON_ARCHITECTURE $ARCHITECTURE_CONFIG model.conv2:sparsity=0.8 model.conv1:sparsity=0.0;
#    $PYTHON_DEFAULT $ADMM_CONFIG admm_trainer.main_iterations=10000 admm_trainer.admm_iterations=200"
#)

adv_attack_selection

declare -a CHANGES=(
#  "$PYTHON_DEFAULT $ANALYZER_CONFIG adv_settings.adv_attack_selection=0 adv_settings.adv_save_enabled=true \
#  adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=1000;
#  $PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.8"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=1.5"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=3.0"

# ======= PGD
#  "$PYTHON_DEFAULT $ANALYZER_CONFIG adv_settings.adv_attack_selection=1 adv_settings.adv_save_enabled=true \
#  adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=10000;
#  $PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=0.0313725490196078"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=0.0627450980392157"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=0.125490196078431"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=0.2509803921568627"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=0.5019607843137255"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG PGD:eps=1.0"


# ====== JSMA
#  "$PYTHON_DEFAULT $ANALYZER_CONFIG adv_settings.adv_attack_selection=2 adv_settings.adv_save_enabled=true \
#  adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=10000;
#  $PYTHON_ATTACKS $ATTACKS_CONFIG JSMA:theta=1.0 JSMA:gamma=0.03"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG JSMA:theta=1.0 JSMA:gamma=0.06"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG JSMA:theta=1.0 JSMA:gamma=0.1"
#  "$PYTHON_ATTACKS $ATTACKS_CONFIG JSMA:theta=1.0 JSMA:gamma=0.2"

  "$PYTHON_DEFAULT $ANALYZER_CONFIG adv_settings.adv_attack_selection=0 adv_settings.adv_save_enabled=true \
  adv_settings.adv_original_save_enabled=true adv_settings.adv_sample_range_start=0 adv_settings.adv_sample_range_end=10000;
  $PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.1"
  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.5"
  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.7"
  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.8"
  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=0.9"
  "$PYTHON_ATTACKS $ATTACKS_CONFIG DeepFool:steps=100 DeepFool:overshoot=1.0"
)

# Anzahl der Testläufe
NUM_RUNS=${#CHANGES[@]}

for (( i=0; i<$NUM_RUNS; i++ )); do
    RUN_SAVE_PATH="${BASE_SAVE_PATH}/run_${i}"
    mkdir -p $RUN_SAVE_PATH

    # Ändere die Konfigurationsdatei für den aktuellen Testrun
    echo "Aktualisiere Konfigurationsdatei für Testrun $((i))..."
    eval "${CHANGES[$i]}; $PYTHON_DEFAULT $ANALYZER_CONFIG save_path=$RUN_SAVE_PATH"

    if [ $? -ne 0 ]; then
        echo "Fehler beim Aktualisieren der Konfigurationsdatei nach Testrun $((i))!"
        exit 1
    fi

    echo "Starte Testrun $((i))..."

    # Führe den Testlauf durch
    $MAIN_COMMAND

    if [ $? -ne 0 ]; then
        echo "Testrun $((i)) fehlgeschlagen!"
        exit 1
    fi

    echo "Testrun $((i)) abgeschlossen."
    echo "Konfigurationsdatei nach Testrun $((i)) erfolgreich aktualisiert."
done

echo "Alle Testläufe erfolgreich abgeschlossen."

