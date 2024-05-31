#!/bin/bash

TRAINER_CONFIG="configs/TrainerConfig.json"
DATAHANDLER_CONFIG="configs/DataHandlerConfig.json"
ARCHITECTURE_CONFIG="configs/preOptimizingTuning/ADMModelArchitecture.json"
ADMM_CONFIG="configs/preOptimizingTuning/ADMMConfig.json"
ANALYZER_CONFIG="configs/analyzer/AnalyzerConfig.json"
PYTHON_DEFAULT="python change_config.py"
PYTHON_ARCHITECTURE="python change_architecture_config.py"
MAIN_COMMAND="python main.py"  # Der Befehl zum Ausführen der Tests


# Liste der Änderungen nach jedem Testlauf
declare -a CHANGES=(
    "$PYTHON_ARCHITECTURE $ARCHITECTURE_CONFIG model.conv1:sparsity=null model.layer1.0.conv1:sparsity=0.0;
    $PYTHON_DEFAULT $ADMM_CONFIG admm_trainer.main_iterations=100000 admm_trainer.admm_iterations=200"
)

# Anzahl der Testläufe
NUM_RUNS=${#CHANGES[@]}


for (( i=0; i<$NUM_RUNS; i++ )); do
    echo "Starte Testlauf $((i+1))..."

    # Führe den Testlauf durch
    #$TEST_COMMAND

#    if [ $? -ne 0 ]; then
#        echo "Testlauf $((i+1)) fehlgeschlagen!"
#        exit 1
#    fi

    echo "Testlauf $((i+1)) abgeschlossen."

    # Ändere die Konfigurationsdatei
    echo "Aktualisiere Konfigurationsdatei..."
    eval ${CHANGES[$i]}

    if [ $? -ne 0 ]; then
        echo "Fehler beim Aktualisieren der Konfigurationsdatei nach Testlauf $((i+1))!"
        exit 1
    fi

    echo "Konfigurationsdatei nach Testlauf $((i+1)) erfolgreich aktualisiert."
done

echo "Alle Testläufe erfolgreich abgeschlossen."

