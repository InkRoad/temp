$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Get-Command python -ErrorAction SilentlyContinue
$mistFile = Join-Path $projectRoot "SatEdgeSim\settings\locationflie\edge_devices\mist Fixed Position.csv"
$firstLine = ""
if (Test-Path $mistFile) {
    $firstLine = Get-Content $mistFile -TotalCount 1
}

if ($firstLine -like "version https://git-lfs.github.com/spec/*") {
    if ($null -eq $python) {
        throw "Python was not found. Install Python or run tools\generate_mist_positions.py manually."
    }
    & $python.Source tools\generate_mist_positions.py
}

$mvnPath = $null
$mvn = Get-Command mvn.cmd -ErrorAction SilentlyContinue
if ($null -ne $mvn) {
    $mvnPath = $mvn.Source
}
if ($null -eq $mvnPath) {
    $ideaMaven = "D:\Program Files\IntelliJ IDEA 2025.2.4\plugins\maven\lib\maven3\bin\mvn.cmd"
    if (Test-Path $ideaMaven) {
        $mvnPath = $ideaMaven
    }
}
if ($null -eq $mvnPath) {
    throw "Maven was not found. Install Maven or update run_chapter4.ps1 with your mvn.cmd path."
}

& $mvnPath exec:java "-Dexec.mainClass=edu.weijunyong.satedgesim.MainApplication" "-Dexec.args=chapter4"

if ($null -ne $python) {
    & $python.Source tools\extract_chapter4_results.py
    & $python.Source tools\create_chapter4_speed_results.py
    & $python.Source tools\plot_chapter4_results.py
    & $python.Source tools\extract_thesis_chapter4_figures.py
}
