$scriptFolder = "\\host.lan\Data"
$pythonScriptFile = "$scriptFolder\server\main.py"
$pythonServerPort = 5000

# Resolve the python executable — prefer PATH, fall back to known install location
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
}

# Start the flask computer use server
Write-Host "Running the server on port $pythonServerPort using $pythonExe"
& $pythonExe $pythonScriptFile --port $pythonServerPort
