# Pattern Pruning Framework for LeNet and ResNet18 with ADMM Support

This repository provides a framework developed as part of a thesis project to apply pattern pruning with support from ADMM (Alternating Direction Method of Multipliers) on LeNet and ResNet18 models. The framework includes various evaluation methods for benchmarking the performance of the models post-training.

## Getting Started

To execute the `main.py` script from the project directory via the CLI, use the following command:

```
python main.py --model <model_name> [options]
```

### CLI Arguments

- `--model`: Specifies the model architecture to be used (e.g., `LeNet`, `ResNet18`).
- `--model_path`: Path to the pre-trained model weights.
- `--cuda_enabled`: Enables CUDA support for GPU acceleration.
- `--disable_analysis_training`: Disables additional analysis during training.
- `--adv_training`: Enables adversarial training mode.
- `--adv_training_ratio`: Ratio for adversarial training if enabled (e.g., `0.1`).
- `--optimization_testing_monitoring_enable`: Enables optimization and monitoring tests during training.
- `--create_architecture_config`: Generates a template for the architecture configuration file for pattern pruning.

### Example Configurations

Two example configurations for LeNet and ResNet18 models are provided in the `config-<model>` folders. To run the corresponding model, rename the folder to `configs` and use the following CLI commands.

#### LeNet Example
This example uses CUDA, activates adversarial training, and demonstrates trivial pattern pruning and connectivity pruning.

```bash
python main.py --model LeNet --model_path LeNet_admm_train.pth --cuda_enabled --disable_analysis_training --adv_training --adv_training_ratio 0.1 --optimization_testing_monitoring_enable
```

#### ResNet18 Example
This example uses CUDA and demonstrates SCP pattern pruning.

```bash
python main.py --model ResNet18 --cuda_enabled --disable_analysis_training --optimization_testing_monitoring_enable
```

### Pattern Pruning Configuration

Pattern pruning is parameterized through the config files. Each model configuration folder contains:

- Adversarial test settings
- Dataset arguments
- Default trainer settings
- ADMM trainer configurations

To extend the framework to other models, the source code must be modified accordingly. Due to time constraints during the thesis project, further code simplifications were not possible.

### Platform Support

These examples are tested on Windows. To run the framework on Unix-based systems, modify the input and config file paths to use Unix notation.

### Architecture Export

Using the `--create_architecture_config` flag generates a template for the model architecture config. This config specifies which layers to prune and the pruning rate. The file should be saved in `configs/PreOptimizingTuning/`.