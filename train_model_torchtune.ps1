param (
    [int]$Epochs,
    [double]$LearningRate,
    [string]$TrainData,
    [string]$BaseModel = "Meta-llama/Llama-3.2-1B-Instruct",
    [int]$LoraRank,
    [int]$LoraAlpha,
    [double]$LoraDropout,
    [int]$MaxSeqLength,
    [int]$WarmupSteps,
    [int]$Seed,
    [string]$SchedulerType,
    [int]$BatchSize,
    [string]$OutputDir,
    [string]$Quantization = "Q4_K_M", # Default quantization value
    [double]$WeightDecay,
    [switch]$UseCheckpoint,
    [string]$HfToken,
    [switch]$FastTransfer, # For non-AMD mode: enables fast transfer via HF_HUB_ENABLE_HF_TRANSFER
    [string]$GpuArch = ""          # If using an AMD GPU, specify e.g. "gfx90a". Leave empty for default mode.
)

# Log received parameters
Write-Host "Parameters passed:" -ForegroundColor Cyan
if ($Epochs) { Write-Host "Epochs: $Epochs" }
if ($LearningRate) { Write-Host "LearningRate: $LearningRate" }
if ($TrainData) { Write-Host "TrainData: $TrainData" }
if ($BaseModel) { Write-Host "BaseModel: $BaseModel" }
if ($LoraRank) { Write-Host "LoraRank: $LoraRank" }
if ($LoraAlpha) { Write-Host "LoraAlpha: $LoraAlpha" }
if ($LoraDropout -ne $null) { Write-Host "LoraDropout: $LoraDropout" }
if ($MaxSeqLength) { Write-Host "MaxSeqLength: $MaxSeqLength" }
if ($WarmupSteps) { Write-Host "WarmupSteps: $WarmupSteps" }
if ($Seed) { Write-Host "Seed: $Seed" }
if ($SchedulerType) { Write-Host "SchedulerType: $SchedulerType" }
if ($BatchSize) { Write-Host "BatchSize: $BatchSize" }
if ($OutputDir) { Write-Host "OutputDir: $OutputDir" } else { $OutputDir = "outputs" }
if ($Quantization) { Write-Host "Quantization: $Quantization" }
if ($WeightDecay) { Write-Host "WeightDecay: $WeightDecay" }
if ($UseCheckpoint) { Write-Host "UseCheckpoint: Enabled" } else { Write-Host "UseCheckpoint: Disabled" }

# Log GPU mode or fast transfer based on parameters
if ($GpuArch -and $GpuArch -ne "") {
    Write-Host "GPU Architecture: $GpuArch" -ForegroundColor Cyan
}
else {
    if ($FastTransfer) {
        Write-Host "FastTransfer: Enabled (HF_HUB_ENABLE_HF_TRANSFER=1)" -ForegroundColor Cyan
    }
    else {
        Write-Host "FastTransfer: Disabled (HF_HUB_ENABLE_HF_TRANSFER=0)" -ForegroundColor Cyan
    }
}

# Define the Docker container name and check if it is running
$ContainerName = "kolo_container"
$containerRunning = docker ps --format "{{.Names}}" | Select-String -Pattern $ContainerName
if (-not $containerRunning) {
    Write-Host "Error: Container '$ContainerName' is not running." -ForegroundColor Red
    exit 1
}

# --- Define BaseModel to config mapping ---
$configMap = @{
    "Meta-llama/Llama-3.1-8B-Instruct"   = "/app/torchtune/configs/llama3_1/8B_qlora_single_device.yaml"
    "Meta-llama/Llama-3.2-3B-Instruct"   = "/app/torchtune/configs/llama3_2/3B_qlora_single_device.yaml"
    "Meta-llama/Llama-3.2-1B-Instruct"   = "/app/torchtune/configs/llama3_2/1B_qlora_single_device.yaml"
    "mistralai/Mistral-7B-Instruct-v0.1" = "/app/torchtune/configs/mistralai/7B_qlora_single_device.yaml"
}

# Retrieve the configuration value based on the provided BaseModel.
if ($configMap.ContainsKey($BaseModel)) {
    $configValue = $configMap[$BaseModel]
}
else {
    throw "Error: The specified BaseModel '$BaseModel' was not found in the configuration mapping."
}

Write-Host "Using configuration: $configValue for BaseModel: $BaseModel" -ForegroundColor Cyan

# --- Begin BaseModel download step ---
if (-not $HfToken) {
    Write-Host "Error: Hugging Face token must be provided." -ForegroundColor Red
    exit 1
}

$hfTransferValue = if ($FastTransfer) { "1" } else { "0" }
$downloadCommand = "export HF_HUB_ENABLE_HF_TRANSFER=$hfTransferValue && source /opt/conda/bin/activate kolo_env && tune download $BaseModel --ignore-patterns 'original/consolidated.00.pth' --hf-token '$HfToken'"

Write-Host "Downloading BaseModel using command:" -ForegroundColor Yellow
Write-Host $downloadCommand -ForegroundColor Yellow

try {
    docker exec -it $ContainerName /bin/bash -c $downloadCommand
    if ($?) {
        Write-Host "BaseModel downloaded successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to download BaseModel." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred during BaseModel download: $_" -ForegroundColor Red
    exit 1
}

# --- Begin torchtune run ---
# Build the base torchtune command string using the configuration from the mapping.
if ($GpuArch -and $GpuArch -ne "") {
    # AMD GPU branch: set HIP alloc conf and ROCm arch.
    $command = "export PYTORCH_HIP_ALLOC_CONF='garbage_collection_threshold:0.8,max_split_size_mb:512' && PYTORCH_ROCM_ARCH=$GpuArch source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $configValue"
}
else {
    # Default branch.
    $command = "source /opt/conda/bin/activate kolo_env && tune run lora_finetune_single_device --config $configValue"
}

# Append dynamic parameters with defaults
if ($Epochs) {
    $command += " epochs=$Epochs"
}
else {
    $command += " epochs=3"
}

if ($BatchSize) {
    $command += " batch_size=$BatchSize"
}
else {
    $command += " batch_size=1"
}

if ($TrainData) {
    $command += " dataset.data_files='$TrainData'"
}
else {
    if ($BaseModel -like "*mistral*") {
        $command += " dataset.data_files=./data_alpaca.json"
    }
    else {
        $command += " dataset.data_files=./data_chat.json"
    }
}


if ($BaseModel -like "*mistral*") {
    # Mistral: alpaca dataset ONLY
    $command += " dataset._component_=torchtune.datasets.alpaca_dataset"
    $command += " dataset.source=json"
}
else {
    # Llama: chat dataset
    $command += " dataset._component_=torchtune.datasets.chat_dataset"
    $command += " dataset.source=json"
    $command += " dataset.conversation_column=conversations"
    $command += " dataset.conversation_style=sharegpt"
}

if ($BaseModel -like "*mistral*") {
    Write-Host "INFO: Mistral detected → Alpaca inference (raw prompt)" -ForegroundColor Cyan
}
else {
    Write-Host "INFO: Llama detected → Chat inference (role-based)" -ForegroundColor Cyan
}


if ($LoraRank) {
    $command += " model.lora_rank=$LoraRank"
}
else {
    $command += " model.lora_rank=16"
}

if ($LoraAlpha) {
    $command += " model.lora_alpha=$LoraAlpha"
}
else {
    $command += " model.lora_alpha=16"
}

if ($LoraDropout -ne $null) {
    $command += " model.lora_dropout=$LoraDropout"
}

if ($LearningRate) {
    $command += " optimizer.lr=$LearningRate"
}
else {
    $command += " optimizer.lr=1e-4"
}

if ($MaxSeqLength) {
    $command += " tokenizer.max_seq_len=$MaxSeqLength"
}

if ($WarmupSteps) {
    $command += " lr_scheduler.num_warmup_steps=$WarmupSteps"
}
else {
    $command += " lr_scheduler.num_warmup_steps=100"
}

if ($Seed) {
    $command += " seed=$Seed"
}

if ($SchedulerType) {
    $command += " lr_scheduler._component_=torchtune.training.lr_schedulers.get_${SchedulerType}_schedule_with_warmup"
}
else {
    $command += " lr_scheduler._component_=torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup"
}

if ($WeightDecay) {
    $command += " optimizer.weight_decay=$WeightDecay"
}
else {
    $command += " optimizer.weight_decay=0.01"
}

if ($UseCheckpoint) {
    $command += " resume_from_checkpoint=True"
}
else {
    $command += " resume_from_checkpoint=False"
}

# Set the output directory; default is "outputs"
$FullOutputDir = "/var/kolo_data/torchtune/$OutputDir"
$command += " output_dir='$FullOutputDir'"

# Log a note on quantization if provided
if ($Quantization) {
    Write-Host "Note: Quantization parameter '$Quantization' is provided and will be used for quantization."
}

Write-Host "Executing torchtune command inside container '$ContainerName':" -ForegroundColor Yellow
Write-Host $command -ForegroundColor Yellow

try {
    docker exec -it $ContainerName /bin/bash -c $command
    if ($?) {
        Write-Host "Torchtune run completed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to execute torchtune run." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred during torchtune run: $_" -ForegroundColor Red
    exit 1
}

# --- Begin post-run merging steps ---
$findEpochCmd = "ls -d ${FullOutputDir}/epoch_* 2>/dev/null | sort -V | tail -n 1"
try {
    $epochFolder = docker exec $ContainerName /bin/bash -c $findEpochCmd
    $epochFolder = $epochFolder.Trim()
    if (-not $epochFolder) {
        Write-Host "Error: No epoch folder found in $FullOutputDir" -ForegroundColor Red
        exit 1
    }
    else {
        Write-Host "Identified epoch folder: $epochFolder" -ForegroundColor Green
    }
}
catch {
    Write-Host "An error occurred while finding the epoch folder: $_" -ForegroundColor Red
    exit 1
}

$mergedModelPath = "${FullOutputDir}/merged_model"
$pythonCommand = "source /opt/conda/bin/activate kolo_env && python /app/merge_lora.py --lora_model '$epochFolder' --merged_model '$mergedModelPath'"
if ($Quantization) {
    $pythonCommand += " --quantization '$Quantization'"
}
Write-Host "Executing merge command inside container '$ContainerName':" -ForegroundColor Yellow
Write-Host $pythonCommand -ForegroundColor Yellow

try {
    docker exec -it $ContainerName /bin/bash -c $pythonCommand
    if ($?) {
        Write-Host "Merge script executed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to execute merge script." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred while executing the merge script: $_" -ForegroundColor Red
    exit 1
}

$conversionCommand = "source /opt/conda/bin/activate kolo_env && /app/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile '$FullOutputDir/Merged.gguf' '$mergedModelPath'"
Write-Host "Executing conversion command inside container '$ContainerName':" -ForegroundColor Yellow
Write-Host $conversionCommand -ForegroundColor Yellow

try {
    docker exec -it $ContainerName /bin/bash -c $conversionCommand
    if ($?) {
        Write-Host "Conversion script executed successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to execute conversion script." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred while executing the conversion script: $_" -ForegroundColor Red
    exit 1
}

# --- Begin quantization step ---
if (-not $Quantization) {
    Write-Host "Quantization parameter not provided. Skipping quantization step." -ForegroundColor Yellow
}
else {
    $quantUpper = $Quantization.ToUpper()
    $quantizeCommand = "source /opt/conda/bin/activate kolo_env && /app/llama.cpp/llama-quantize '$FullOutputDir/Merged.gguf' '$FullOutputDir/Merged${Quantization}.gguf' $quantUpper"
    Write-Host "Executing quantization command inside container '$ContainerName':" -ForegroundColor Yellow
    Write-Host $quantizeCommand -ForegroundColor Yellow

    try {
        docker exec -it $ContainerName /bin/bash -c $quantizeCommand
        if ($?) {
            Write-Host "Quantization script executed successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "Failed to execute quantization script." -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Host "An error occurred while executing the quantization script: $_" -ForegroundColor Red
        exit 1
    }
}
