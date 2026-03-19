Param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 3000
)

Write-Host "[EntropyHub] API ve Frontend demo ortami baslatiliyor..."

$repoRoot = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$repoRoot'; py -m uvicorn api.main:app --host $ApiHost --port $ApiPort --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$repoRoot\\frontend'; npm start -- --port $FrontendPort"

Write-Host "[EntropyHub] API: http://$ApiHost`:$ApiPort"
Write-Host "[EntropyHub] Frontend: http://localhost:$FrontendPort"
Write-Host "[EntropyHub] Swagger: http://$ApiHost`:$ApiPort/docs"
