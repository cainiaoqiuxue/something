@echo off
chcp 65001
cls
echo 首先确认input_dir已经得到更新
pause
cls
call activate base
python tw_pipeline/main_bash.py
echo 运行完成
pause