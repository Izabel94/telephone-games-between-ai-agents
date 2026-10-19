@echo off
setlocal
cd /d "%~dp0"
echo.
echo === Telephone workshop: setup (Windows) ===
echo.

rem Windows cannot install files with paths over 260 characters; the deepest file in .venv needs 162 of them.
if not "%CD:~90,1%"=="" goto :toolong

call :findpython
if not defined PY call :getpython
if not defined PY goto :failed
echo Using Python:
%PY% --version

if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto :failed

set /a TRIES=0
:install
set /a TRIES+=1
echo.
echo Installing packages. The first time this takes a few minutes...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt && goto :key
if %TRIES% GEQ 3 goto :failed
echo Install hiccup, trying again (%TRIES% of 3)...
ping -n 6 127.0.0.1 >nul
goto :install

:key
if exist ".env" goto :check
if defined GEMINI_API_KEY goto :writekey
echo.
echo Paste your Gemini API key here (right-click or Ctrl+V), then press Enter:
set /p "GEMINI_API_KEY="
if not defined GEMINI_API_KEY goto :nokey
:writekey
> ".env" echo GEMINI_API_KEY=%GEMINI_API_KEY%

:check
echo.
echo Checking that the models answer...
".venv\Scripts\python.exe" check_models.py || goto :failed
echo.
echo === Setup done. Now double-click Start-Windows to open the notebook. ===
goto :end

:findpython
set "PY="
set "CHECK=import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))"
for %%V in (3.12 3.13 3.14) do py -%%V -c "%CHECK%" >nul 2>&1 && set "PY=py -%%V" && exit /b 0
for %%P in (python python3) do %%P -c "%CHECK%" >nul 2>&1 && set "PY=%%P" && exit /b 0
for %%V in (312 313 314) do if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set PY="%LocalAppData%\Programs\Python\Python%%V\python.exe"&& exit /b 0
for %%V in (312 313 314) do if exist "%ProgramFiles%\Python%%V\python.exe" set PY="%ProgramFiles%\Python%%V\python.exe"&& exit /b 0
exit /b 0

:getpython
echo Python 3.12, 3.13 or 3.14 was not found. Installing Python 3.12 into this folder...
set "UV_UNMANAGED_INSTALL=%CD%\.uv"
set "UV_PYTHON_INSTALL_DIR=%CD%\.python"
set "UV_CACHE_DIR=%CD%\.uv\cache"
if not exist ".uv\uv.exe" powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://github.com/astral-sh/uv/releases/download/0.12.19/uv-installer.ps1 | iex"
if not exist ".uv\uv.exe" exit /b 0
.uv\uv.exe python install 3.12 --no-bin --no-registry || exit /b 0
for /f "delims=" %%P in ('.uv\uv.exe python find 3.12 --managed-python') do set PY="%%P"
exit /b 0

:toolong
echo This folder is too deep for Windows to install into:
echo "%CD%"
echo.
echo Please delete this folder, then unzip tw.zip into your user folder:
echo in File Explorer, type %%USERPROFILE%% in the address bar, press Enter, and extract tw.zip there.
echo Then double-click Setup-Windows again.
goto :end

:nokey
echo No key entered. Double-click Setup-Windows again when you have your key.
goto :end

:failed
echo.
echo Something went wrong. Please send a screenshot of this window to the workshop organisers.

:end
echo.
if not defined CI pause
endlocal
