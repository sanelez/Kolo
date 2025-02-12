# Kolo

**Kolo** is a lightweight tool designed for **fast and efficient fine-tuning and testing of Large Language Models (LLMs)** on your local machine. It leverages cutting-edge tools to simplify the fine-tuning process, making it as quick and seamless as possible.

## 🚀 Features

- 🏗 **Lightweight**: Minimal dependencies, optimized for speed.
- ⚡ **Runs Locally**: No need for cloud-based services; fine-tune models on your own machine.
- 🛠 **Easy Setup**: Simple installation and execution with Docker.
- 🔌 **Support for Popular Frameworks**: Integrates with major LLM toolkits.

## 🛠 Tools Used

Kolo is built using a powerful stack of LLM tools:

- [Unsloth](https://github.com/unslothai/unsloth) – Open-source LLM fine-tuning; faster training, lower VRAM.
- [Torchtune](https://github.com/pytorch/torchtune) – Native PyTorch library simplifying LLM fine-tuning workflows.
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) – Fast C/C++ inference for Llama models.
- [Ollama](https://ollama.ai/) – Portable, user-friendly LLM model management and deployment.
- [Docker](https://www.docker.com/) – Containerized environment ensuring consistent, scalable deployments.
- [Open WebUI](https://github.com/open-webui/open-webui) – Intuitive self-hosted web interface for LLM management.

## System Requirements

- Windows 10 OS or higher. Might work on Linux & Mac (Untested)
- Nvidia GPU with CUDA 12.1 capability and 8GB+ of VRAM
- 16GB+ System RAM

## Issues or Feedback

Join our [Discord group](https://discord.gg/Ewe4hf5x3n)!

## 🏃 Getting Started

### 1️⃣ Install Dependencies

Ensure [HyperV](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v?pivots=windows) is installed.

Ensure [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) is installed; alternative [guide](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers).

Ensure [Docker Desktop](https://docs.docker.com/get-docker/) is installed.

### 2️⃣ Build the Image

```bash
./build_image.ps1
```

### 3️⃣ Run the Container

If running for first time:

```bash
./create_and_run_container.ps1
```

For subsequent runs:

```bash
./run_container.ps1
```

### 4️⃣ Copy Training Data

```bash
./copy_training_data.ps1 -f examples/God.jsonl -d data.jsonl
```

### 5️⃣ Train Model

#### Using Unsloth

```bash
./train_model_unsloth.ps1 -OutputDir "GodOutput" -Quantization "Q4_K_M" -TrainData "data.jsonl"
```

All available parameters

```bash
./train_model_unsloth.ps1 -Epochs 3 -LearningRate 1e-4 -TrainData "data.jsonl" -BaseModel "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" -ChatTemplate "llama-3.1" -LoraRank 16 -LoraAlpha 16 -LoraDropout 0 -MaxSeqLength 1024 -WarmupSteps 10 -SaveSteps 500 -SaveTotalLimit 5 -Seed 1337 -SchedulerType "linear" -BatchSize 2 -OutputDir "GodOutput" -Quantization "Q4_K_M" -WeightDecay 0
```

#### Using Torchtune

Requirements: Create a [Hugging Face](https://huggingface.co/) account and create a token. You will also need to get permission from Meta to use their models. Search the Base Model name on Hugging Face website and get access before training.

```bash
./train_model_torchtune.ps1 -OutputDir "GodOutput" -Quantization "Q4_K_M" -TrainData "data.json" -HfToken "your_token"
```

All available parameters

```bash
./train_model_torchtune.ps1 -HfToken "your_token" -Epochs 3 -LearningRate 1e-4 -TrainData "data.json" -BaseModel "Meta-llama/Llama-3.2-1B-Instruct" -LoraRank 16 -LoraAlpha 16 -LoraDropout 0 -MaxSeqLength 1024 -WarmupSteps 10 -Seed 1337 -SchedulerType "cosine" -BatchSize 2 -OutputDir "GodOutput" -Quantization "Q4_K_M" -WeightDecay 0
```

For more information about fine tuning parameters please refer to the [Fine Tune Training Guide](https://github.com/MaxHastings/Kolo/blob/main/FineTuningGuide.md).

### 6️⃣ Install Model

#### Using Unsloth

```bash
./install_model.ps1 "God" -Tool "unsloth" -OutputDir "GodOutput" -Quantization "Q4_K_M"
```

#### Using Torchtune

```bash
./install_model.ps1 "God" -Tool "torchtune" -OutputDir "GodOutput" -Quantization "Q4_K_M"
```

### 7️⃣ Test Model

Open your browser and navigate to [localhost:8080](http://localhost:8080/)

### Other Commands

```bash
./uninstall_model.ps1 "God"
```

```bash
./list_models.ps1
```

```bash
./delete_model.ps1 "GodOutput" -Tool "unsloth|torchtune"
```

```bash
./copy_scripts.ps1
```

```bash
./copy_configs.ps1
```

## 🔧 Advanced Users

### SSH Access

To quickly SSH into the Kolo container for installing additional tools or running scripts directly:

```bash
./connect.ps1
```

If prompted for a password, use:

```bash
password 123
```

Alternatively, you can connect manually via SSH:

```bash
ssh root@localhost -p 2222
```

### WinSCP (SFTP Access)

You can use [WinSCP](https://winscp.net/eng/index.php) or any other SFTP file manager to access the Kolo container’s file system. This allows you to manage, modify, add, or remove scripts and files easily.

Connection Details:

- Host: localhost
- Port: 2222
- Username: root
- Password: 123

This setup ensures you can easily transfer files between your local machine and the container.
