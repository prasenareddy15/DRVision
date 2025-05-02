@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing pip-tools...
pip install pip-tools

echo Compiling requirements.txt...
pip-compile requirements.in

echo Installing all dependencies...
pip install -r requirements.txt

echo ✅ Setup complete! Use "venv\Scripts\activate" to activate the environment.
pause
