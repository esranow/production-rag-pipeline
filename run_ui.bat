@echo off
echo ========================================================
echo  Starting Production RAG Pipeline Web UI...
echo ========================================================
cd /d %~dp0
streamlit run app_ui.py --server.port 8501
pause
