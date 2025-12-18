# Example usage:
# ./copy_training_data.ps1 -f "C:\path\to\file.jsonl" -d "data.jsonl"

param (
    [string]$f, # Local file path
    [string]$d = "data.jsonl"       # Destination filename inside the container (default: data.jsonl)
)

# Check if the file exists locally
if (-Not (Test-Path $f)) {
    Write-Host "Error: File does not exist at path: $f" -ForegroundColor Red
    exit 1
}

# Step 1: Copy the file into the container
try {
    Write-Host "Copying $f to container kolo_container at /app/$d..."
    docker cp $f "kolo_container`:/app/$d"
    
    if ($?) {
        Write-Host "File copied successfully as $d!" -ForegroundColor Green
    }
    else {
        Write-Host "Failed to copy file." -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "An error occurred during copy: $_" -ForegroundColor Red
    exit 1
}

$jsonOutputFile = "data.json"

try {
    Write-Host "Running conversion script in container kolo_container..."
    docker exec kolo_container bash -lc "source /opt/conda/bin/activate kolo_env && python /app/convert_jsonl_to_json.py /app/$d --chat_out /app/data_chat.json --alpaca_out /app/data_alpaca.json"

    if ($?) {
        Write-Host "Conversion successful! Converted file created as $jsonOutputFile in the container." -ForegroundColor Green
    }
    else {
        Write-Host "Failed to run conversion script." -ForegroundColor Red
    }
}
catch {
    Write-Host "An error occurred while running the conversion script: $_" -ForegroundColor Red
}

