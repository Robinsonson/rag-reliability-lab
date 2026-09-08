# Launch the legacy Streamlit chat with the maintained Python 3.12 environment.
$env:STREAMLIT_SERVER_FILE_WATCHER_TYPE = "none"
Set-Location $PSScriptRoot
.\.venv312\Scripts\streamlit.exe run app.py --server.fileWatcherType none
