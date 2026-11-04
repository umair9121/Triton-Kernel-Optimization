# Render report tabs to PNG with headless Edge. Serve the HTML first, e.g.:
#   python -m http.server 8765 --bind 127.0.0.1 --directory reports
param(
    [string]$Report = "matmul_kernel",
    [string[]]$Tabs = @("overview", "layouts", "shared", "global", "tiling", "ir"),
    [string]$OutDir = "docs/images",
    [string]$Prefix = "",
    [int]$Port = 8765
)
$edge = @("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
          "C:\Program Files\Microsoft\Edge\Application\msedge.exe") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
New-Item -ItemType Directory -Force $OutDir | Out-Null
$out = (Resolve-Path $OutDir).Path
if (-not $Prefix) { $Prefix = $Report }
foreach ($tab in $Tabs) {
    & $edge --headless=new --disable-gpu --hide-scrollbars --window-size=1440,1000 `
        --virtual-time-budget=3000 "--screenshot=$out\$Prefix`_$tab.png" `
        "http://127.0.0.1:$Port/$Report.html#$tab" 2>$null
    Start-Sleep -Milliseconds 1500
}
Get-ChildItem $out -Filter "$Prefix*.png" | Select-Object Name, Length
