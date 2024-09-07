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

    