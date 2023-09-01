# -*- coding:utf-8 -*-

"""
 FileName     : main_bash.py
 Type         : pyspark/pysql/python
 Arguments    : None
 Author       : xingyuanfan@tencent.com
 Date         : 2023-07-25
 Description  : 
"""
import argparse
import os
import yaml

with open('tw_pipeline/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg = argparse.Namespace(**cfg)
raw_dir = os.path.join(cfg.project_root, cfg.raw_data_dir)
output_name = os.listdir(raw_dir)[0].split('_')[0]
output_name += '_2021_01_01_2023_06_30'
print('默认项目文件名: {}'.format(output_name))
choose = input('[y: 确认 n: 自定义项目名]: ')
if choose == 'n':
    output_name = input('自定义项目名: ')

os.system("cls")

length = len(output_name) + 6
logo = '''
{}
## {} ##
{}
'''.format('#' * length, output_name, '#' * length)
print(logo)
cfg.output_name = output_name

from tw_bash import TWPipeline
tw = TWPipeline(cfg)
tw.run()