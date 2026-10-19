@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\jupyter.exe" goto :nosetup
echo Opening the notebook in your browser...
echo Keep this window open while you work. To stop, close the browser tab and this window.
".venv\Scripts\jupyter.exe" lab workshop.ipynb %*
goto :end

:nosetup
echo Setup has not been run yet. Double-click Setup-Windows first.
if not defined CI pause

:end
endlocal
