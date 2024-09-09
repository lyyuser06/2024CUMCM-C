import pandas as pd

def readFromExcel(file_path, sheet_name) : 
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df
def displayExcel(dataframe) :
    print(dataframe)

def readOriginalData() :
    path_appendix_1 = "CUMCM2024Problems\\C题\\附件1.xlsx"
    path_appendix_2 = "CUMCM2024Problems\\C题\\附件2.xlsx"

    appendix_1_sheet1 = "乡村的现有耕地"
    appendix_1_sheet2 = "乡村种植的农作物"
    appendix_2_sheet1 = "2023年的农作物种植情况"
    appendix_2_sheet2 = "2023年统计的相关数据"
    
    df_apdx1_s1 = readFromExcel(path_appendix_1, appendix_1_sheet1)
    df_apdx1_s2 = readFromExcel(path_appendix_1, appendix_1_sheet2)
    df_apdx2_s1 = readFromExcel(path_appendix_2, appendix_2_sheet1)
    df_apdx2_s2 = readFromExcel(path_appendix_2, appendix_2_sheet2)
    
    return df_apdx1_s1, df_apdx1_s2, df_apdx2_s1, df_apdx2_s2

def dataDrop(field_data, crop_data) :
    offset = [41, 42, 43, 44]
    crop_data = crop_data.drop(offset, axis=0)
    # print(crop_data)
    
    field_name = field_data['地块名称']
    crop_name = crop_data['作物名称']
    field_type = field_data['地块类型'].unique()
    crop_type = crop_data['作物类型'].unique()
    
    return field_name, crop_name, field_type, crop_type

if __name__ == 'read_data.py' :
    print(field_type)
    
    
'''
print(field_data, crop_data, plant_data_2023, stat_2023)
def dropData() :
    df_apdx1_s1, df_apdx1_s2, df_apdx2_s1, df_apdx2_s2 = readOriginalData()
    df_apdx1_s1 = df_apdx1_s1[]
'''

# 通过excel操作获取的字典: 作物编号 -> 根据2023年销量统计的每种作物的预期销售量

sales_expectation = {
    1: 51300.00,   # 黄豆
    2: 19665.00,   # 黑豆
    3: 20160.00,   # 红豆
    4: 29736.00,   # 绿豆
    5: 8887.50,    # 爬豆
    6: 153756.00,  # 小麦
    7: 119475.00,  # 玉米
    8: 64260.00,   # 谷子
    9: 27000.00,   # 高粱
    10: 11250.00,  # 黍子
    11: 1350.00,   # 荞麦
    12: 31590.00,  # 南瓜
    13: 32400.00,  # 红薯
    14: 12600.00,  # 莜麦
    15: 9000.00,   # 大麦
    16: 18900.00,  # 水稻
    17: 32616.00,  # 豇豆
    18: 24192.00,  # 刀豆
    19: 7560.00,   # 芸豆
    20: 27000.00,  # 土豆
    21: 32589.00,  # 西红柿
    22: 40824.00,  # 茄子
    23: 810.00,    # 菠菜
    24: 2349.00,   # 青椒
    25: 3240.00,   # 菜花
    26: 3645.00,   # 包菜
    27: 4050.00,   # 油麦菜
    28: 31932.00,  # 小青菜
    29: 11745.00,  # 黄瓜
    30: 3780.00,   # 生菜
    31: 1080.00,   # 辣椒
    32: 3240.00,   # 空心菜
    33: 1620.00,   # 黄心菜
    34: 1620.00,   # 芹菜
    35: 135000.00, # 大白菜
    36: 54000.00,  # 白萝卜
    37: 32400.00,  # 红萝卜
    38: 8100.00,   # 榆黄菇
    39: 6480.00,   # 香菇
    40: 16200.00,  # 白灵菇
    41: 3780.00    # 羊肚菌
}


    