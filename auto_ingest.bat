@echo off
:: Activate Conda environment and run the ingestion script
call C:\anaconda\Scripts\activate.bat aq_env
cd /d "D:\ML projects\AQI_Prediction_Engine"
python run_ingestion.py