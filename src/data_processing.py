import glob
import pandas as pd
import numpy as np

# rename Columns
def rename_the_xlsx_files_column(xlsx_name):
    '''
    some files' column names is inconsistent
    '''
    input_data = pd.read_excel(xlsx_name, sheet_name='Sheet1')
    
    if input_data.columns[0] != '时间戳':
        input_data = pd.DataFrame(input_data.values, columns=['时间戳', 'Event_type_眼动形式', 'Average_pupil_diameter_平均瞳孔直径',
           '0_direction_眼跳方向', 'Average_velocity_平均眼跳时间', 'Peak_velocity_眼跳峰值速度',
           '0_amplitude_眼跳幅度', '周期数', '是否发生失误', '执行时间', '反应时间'])
    input_data.to_excel(xlsx_name, sheet_name='Sheet1', index=False)        

# 读取数据
def fill_the_xlsx_files(xlsx_name, has_last_column=False):
    input_data = pd.read_excel(xlsx_name, sheet_name='Sheet1')
    if has_last_column:
        input_data = input_data[:-1]
    # 找到时间戳的最大值
    max_timestamp = input_data['时间戳'].max()

    # 创建新的时间戳序列
    new_timestamps = np.arange(max_timestamp + 20, 30001, 20)

    # 创建一个新的 DataFrame，包含新的时间戳和其他列的复制值
    new_data = pd.DataFrame({
        '时间戳': new_timestamps
    })

    # 复制其他列的值（使用最后一行数据）
    for col in input_data.columns:
        if col != '时间戳':
            new_data[col] = input_data[col].iloc[-1]

    # 将新的数据追加到原始 DataFrame
    input_data = pd.concat([input_data, new_data], ignore_index=True)
    input_data.to_excel(xlsx_name, sheet_name='Sheet1', index=False)