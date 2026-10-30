@echo off
echo Starting STT Voice Data Collection Platform...
start "" http://localhost:8080/index.html
python -m http.server 8080
