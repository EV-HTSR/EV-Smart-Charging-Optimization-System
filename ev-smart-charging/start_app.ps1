Write-Host "Starting EV Charging Application..." -ForegroundColor Green

Write-Host "
1. Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\HARSHIT PATHAK\ev-smart-charging'; .\venv\Scripts\Activate.ps1; python app_backend.py"

Write-Host "
2. Starting Frontend Dashboard..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\HARSHIT PATHAK\ev-smart-charging'; .\venv\Scripts\Activate.ps1; streamlit run app_frontend.py"

Write-Host "
âœ… Both servers starting in separate windows..." -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:8501" -ForegroundColor Cyan
