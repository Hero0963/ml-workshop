@echo off
REM Use PowerShell to get a robust, locale-independent timestamp
for /f %%i in ('powershell -c "Get-Date -Format \"yyyyMMddTHHmmss\""') do set TIMESTAMP=%%i

REM Run pytest and redirect all output to a timestamped text file
pytest > "src\core\tests\reports\test_report_%TIMESTAMP%.txt"
