import read_data, random, math
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import gurobipy as gb

# 阈值：落在0的epsilon邻域则认为近似是0
epsilon = 0.01

# 阈值：每块地种植的最小面积
A_min = 0.1

# 减产率
alpha = 0.25

# 政府收购比例，农民自销比例
gov_acq_rate = 0.6
pes_sale_rate = 0.3

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

field_keys = [  
    "A1", "A2", "A3", "A4", "A5", "A6",  
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14",  
    "C1", "C2", "C3", "C4", "C5", "C6",  
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",  
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16",  
    "F1", "F2", "F3", "F4"  
]  

# 整合数据
field_data, crop_data, plant_data_2023, stat_2023 = read_data.readOriginalData()
field_name, crop_name, field_type, crop_type = read_data.dataDrop(field_data, crop_data)
plant_data_2023['种植地块'] = plant_data_2023['种植地块'].ffill()
stat_2023 = stat_2023.drop([107, 108, 109])
field_area = field_data['地块面积/亩']
df_sale_price = stat_2023

df_sale_price[['Min', 'Max']] = df_sale_price['销售单价/(元/斤)'].str.split('-', expand=True)
df_sale_price['Min'] = pd.to_numeric(df_sale_price['Min'], errors='coerce')
df_sale_price['Max'] = pd.to_numeric(df_sale_price['Max'], errors='coerce')
df_sale_price['Random'] = df_sale_price.apply(lambda row: random.uniform(row['Min'], row['Max']), axis=1)  

# print(df_sale_price)

# 映射：地块类型 -> 地块下标范围 
field_map = {
        '平旱地' : list(range(6)), 
        '梯田' : list(range(6, 20)), 
        '山坡地' : list(range(20, 26)),
        '水浇地' : list(range(26, 34)),
        '普通大棚' : list(range(34, 50)),
        '智慧大棚' : list(range(50, 54))
    }

# print(stat_2023)

# 生成字典，键是地块的名称，值是地块的编号
def field_to_dict(keys):
    field_dict = {}   
    count = 1  
    
    for key in sorted(keys, key=lambda x: (x[0], int(x[1:]))):  
        field_dict[key] = count  
        count += 1  
    return field_dict

# 整合 2023 年种植情况
def gen_data_2023():
    tensor = np.zeros((len(field_name), len(crop_name), 2))

    for index, row in plant_data_2023.iterrows():
        fd_dict = field_to_dict(field_keys)
        i = fd_dict[row['种植地块']] - 1
        j = row['作物编号'] - 1
        s = 1 if row['种植季次'] in ['单季', '第一季'] else 2
        s -= 1
        tensor[i, j, s] += row['种植面积/亩']
        # print(index)
    return tensor

data_2023 = gen_data_2023()

# 引入减产惩罚机制，构建 2023 年罚因子矩阵y[i, j, s]

def generate_matrix_y():
    y = np.zeros((54, 41, 2), dtype=int)
    '''
    for i in range(26):  
        for j in range(15):   
            if data_2023[i, j, 0] > 0 :
                y[i, j, 0] = 1
                y[i, j, 1] = 1
    
    for i in range(26, 34):  
        if data_2023[i, 15, 0] > 0 :
            y[i, 15, 0] = 1
            y[i, 15, 1] = 1
    
    for i in range(50, 54):  
        for j in range(16, 34):  
            if data_2023[i, j, 0] > 0 and data_2023[i, j, 1] > 0:
                y[i, j, 0] = 1
                y[i, j, 1] = 1

    for i in range(50, 54):  
        for j in range(16, 34):  
            if data_2023[i, j, 1] > 0:
                y[i, j, 0] = 1  
    '''
    
    for i in range(54):
        for j in range(41):
            for s in range(2):
                if data_2023[i, j, s] > 0:
                    y[i, j, s] = 1
    return y
                
y_2023 = generate_matrix_y()          
# print(y_2023)

'''
def update_matrix_y(t):
    y = np.zeros((54, 41, 2), dtype=int)
    t -= 2024
    for i in range(26):  
        for j in range(15):   
            if res[i, j, 0, t] > 0 :
                y[i, j, 0] = 1
                y[i, j, 1] = 1
    
    for i in range(26, 34):  
        if res[i, 15, 0, t] > 0 :
            y[i, 15, 0] = 1
            y[i, 15, 1] = 1
    
    for i in range(50, 54):  
        for j in range(16, 34):  
            if res[i, j, 0, t] > 0 and model.x[i, j, 1, t] > 0:
                y[i, j, 0] = 1
                y[i, j, 1] = 1

    for i in range(50, 54):  
        for j in range(16, 34):  
            if model.x[i, j, 1, t] > 0:
                y[i, j, 0] = 1  
    return y
'''

# 生成2023年产量矩阵，在第一个问题中，亩产量以2023年为基准，此后逐年迭代
def generate_matrix_Q_2023():
    Q = np.zeros((54, 41, 2), dtype=float)
    
    for index, row in stat_2023.iterrows():
        i_range = field_map[row['地块类型']]
        j = int(row['作物编号']) - 1
        s = 1 if row['种植季次'] in ['单季', '第一季'] else 2
        s -= 1
        for i in i_range:
            Q[i, j, s] += row['亩产量/斤']  
    return Q

Q = generate_matrix_Q_2023()
# print(Q[26, 18, 0])

# 生成价格向量，在第一问的情境下，我们认为所有种类作物的价格趋近平稳，因此价格取定范围内的值
def generate_matrix_P():
    '''
    P = np.zeros((54, 41, 2), dtype=float)
    
    for index, row in stat_2023.iterrows():
        i_range = field_map[row['地块类型']]
        j = int(row['作物编号']) - 1
        s = 1 if row['种植季次'] in ['单季', '第一季'] else 2
        s -= 1
        for i in i_range:
            P[i, j, s] += row['Max'] 
    '''
    P = np.zeros((41))
    for index, row in stat_2023.iterrows():
        j = int(row['作物编号']) - 1
        P[j] += row['Max']
    return P

P = generate_matrix_P()
# print(P[26, 18, 0])

# 生成成本矩阵，在第一问的情境下，我们认为成本趋近平稳。
def generate_matrix_D():
    D = np.zeros((54, 41, 2), dtype=float)
    
    for index, row in stat_2023.iterrows():
        i_range = field_map[row['地块类型']]
        j = int(row['作物编号']) - 1
        s = 1 if row['种植季次'] in ['单季', '第一季'] else 2
        s -= 1
        for i in i_range:
            D[i, j, s] += row['种植成本/(元/亩)']  
    return D

D = generate_matrix_D()
# print(D[26, 18, 0])

# 定义非线性规划模型
model = pyo.ConcreteModel()

model.I = pyo.RangeSet(1, 54) # 54个地块
model.I_flat = pyo.RangeSet(1, 6) # 平旱地
model.I_step = pyo.RangeSet(7, 20) # 梯田
model.I_hill = pyo.RangeSet(21, 26) # 山坡地
model.I_wet = pyo.RangeSet(27, 34) # 水浇地
model.I_gh = pyo.RangeSet(35, 50) # 普通大棚
model.I_igh = pyo.RangeSet(51, 54) # 智慧大棚

model.J = pyo.RangeSet(1, 41) # 41 种作物
model.J_grn_soya = pyo.RangeSet(1, 5) # 粮食(豆类)
model.J_grn = pyo.RangeSet(6, 16) # 粮食
model.J_rice = pyo.Set(initialize=[16]) # 水稻
model.J_veg_soya = pyo.RangeSet(17, 19) # 蔬菜(豆类)
model.J_veg = pyo.RangeSet(20, 37) # 蔬菜
model.J_veg_usl = pyo.RangeSet(35, 37) # 大白菜, 白萝卜, 红萝卜
model.J_mush = pyo.RangeSet(38, 41) # 食用菌

model.S = pyo.RangeSet(1, 2)    # 季次
model.T = pyo.RangeSet(2024, 2030) # 年份

# 设置决策变量x: 年份t季次s在地块i种植作物j的面积 - 连续变量
model.x = pyo.Var(model.I, model.J, model.S, model.T, domain=pyo.NonNegativeReals)
# 设置罚因子y: 年份t季次s在是否地块i种植作物j - 0-1变量
model.y = pyo.Var(model.I, model.J, model.S, model.T, domain=pyo.Binary)

A = field_area # A 是每个地块的面积

M = 100

# 隐性约束: 当x是正值, 那么y是1, 否则
def xy_relation_rule(model, i, j, s, t):
    return model.x[i, j, s, t] <= M * model.y[i, j, s, t]

model.xy_relation = pyo.Constraint(model.I, model.J, model.S, model.T, rule=xy_relation_rule)

# 约束 (2) 面积
def area_constraint_rule(model, i, j, s, t):
    return sum(model.x[i, j, s, t] for j in model.J) <= A[i - 1]

model.area_constraint = pyo.Constraint(model.I, model.J, model.S, model.T, rule=area_constraint_rule)

model.J_grn_except_rice = (model.J_grn_soya | model.J_grn) - model.J_rice
# print(list(model.J_grain_except_rice))

# 种植耕地限制
# 除水稻外的粮食种植在平旱地,梯田,山坡地
def plant_constraint_1_rule(model, i, j, s, t):
    if i not in model.I_flat | model.I_step | model.I_hill and j in model.J_grn_except_rice:
        return model.x[i, j, s, t] == 0
    return pyo.Constraint.Skip

# 水稻种植在水浇地
def plant_constraint_2_rule(model, i, j, s, t):
    if i not in model.I_wet and j in model.J_rice:
        return model.x[i, j, s, t] == 0
    return pyo.Constraint.Skip

# 除大白菜和两种萝卜之外的限制
model.J_veg_except_usl = (model.J_veg_soya | model.J_veg) - model.J_veg_usl
# print(list(model.J_veg_except_usl))

def plant_constraint_3_rule(model, i, j, s, t):
    cc = (not (i in model.I_wet | model.I_gh | model.I_igh and s==1) 
            and not (i in model.I_igh and s==2))
    if cc and j in model.J_veg_except_usl:
        return model.x[i, j, s, t] == 0
    return pyo.Constraint.Skip

# 大白菜, 两种萝卜种植在水浇地第二季
def plant_constraint_4_rule(model, i, j, s, t):
    if not (i in model.I_wet and s==2) and j in model.J_veg_usl:
        return model.x[i, j, s, t] == 0
    return pyo.Constraint.Skip

# 蘑菇种植在普通大棚第二季
def plant_constraint_5_rule(model, i, j, s, t):
    if not (i in model.I_gh and s==2) and j in model.J_mush:
        return model.x[i, j, s, t] == 0
    return pyo.Constraint.Skip

model.constraint_plant_1 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=plant_constraint_1_rule)
model.constraint_plant_2 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=plant_constraint_2_rule)
model.constraint_plant_3 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=plant_constraint_3_rule)
model.constraint_plant_4 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=plant_constraint_4_rule)
model.constraint_plant_5 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=plant_constraint_5_rule)

# 季次约束: 单季种植的作物(所有粮食类作物)一律只考虑第一季情况, 第二季置为0
# 等价于粮食类作物只能种植在所有地块所有年份的第一季
def season_constraint_rule(model, i, j, s, t):
    if j in model.J_grn | model.J_grn_soya:
        return model.x[i, j, 2, t] == 0
    return pyo.Constraint.Skip

model.season_constraint = pyo.Constraint(model.I, model.J, model.S, model.T, rule=season_constraint_rule)

# 约束 (3) 如果某块水浇地某年第一季种植了水稻那么这块地当年第二季不能种植其他任何作物
def constraint_3_rule(model, i, j, s, t):
    if i in model.I_wet and t in model.T:
        return model.x[i, j, 2, t] <= M * (1 - model.y[i, 16, 1, t]) 
    return pyo.Constraint.Skip

model.constraint_3 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=constraint_3_rule)

# 约束 (4) 轮作约束，每个地块在三年内至少要种植一次豆类作物，并且水浇地第二季只能种植特定的三种作物。
def rotate_constraint_rule(model, i, j, s, t):
    if i in model.I_flat | model.I_step | model.I_hill and t in range(2024, 2030):
        if t==2024:
            return sum(data_2023[i - 1, j - 1, s - 1] for j in model.J_grn_soya for s in model.S) + sum(model.x[i, j, s, t] for j in model.J_grn_soya for s in model.S) + sum(model.x[i, j, s, t + 1] for j in model.J_grn_soya for s in model.S) >= epsilon
        else:
            return sum(model.x[i, j, s, t - 1] for j in model.J_grn_soya for s in model.S) + sum(model.x[i, j, s, t] for j in model.J_grn_soya for s in model.S) + sum(model.x[i, j, s, t + 1] for j in model.J_grn_soya for s in model.S) >= epsilon
    
    elif i in model.I_wet | model.I_gh | model.I_igh and t in range(2024, 2030):
        if t==2024:
            return sum(data_2023[i - 1, j - 1, s - 1] for j in model.J_veg_soya for s in model.S) + sum(model.x[i, j, s, t] for j in model.J_veg_soya for s in model.S) + sum(model.x[i, j, s, t + 1] for j in model.J_veg_soya for s in model.S) >= epsilon
        else:
            return sum(model.x[i, j, s, t - 1] for j in model.J_veg_soya for s in model.S) + sum(model.x[i, j, s, t] for j in model.J_veg_soya for s in model.S) + sum(model.x[i, j, s, t + 1] for j in model.J_veg_soya for s in model.S) >= epsilon
    return pyo.Constraint.Skip

model.rotate_constraint = pyo.Constraint(model.I, model.J, model.S, model.T, rule=rotate_constraint_rule)

# 约束 (5) 种植分散性约束
# 获取 2023 年所有作物中同种种植相距最远的案例值, 作为之后几年的阈值
max_dist = 30

def dsp_constraint_rule(model, i_1, i_2, j, s, t):
    if abs(i_1 - i_2) > max_dist and j in model.J and s in model.S and t in model.T:
        return model.y[i_1, j, s, t] + model.y[i_2, j, s, t] <= 1
    return pyo.Constraint.Skip

model.dsp_constraint = pyo.Constraint(model.I, model.I, model.J, model.S, model.T, rule=dsp_constraint_rule)

# 约束: 种植面积限制，每地块的种植面积不能少于阈值 
# 如果种了某种作物, 面积不能太少

def field_constraint_rule(model, i, j, s, t):
    return sum(model.x[i, j, s, t] for j in model.J) >= A_min

model.field_constraint = pyo.Constraint(model.I, model.J, model.S, model.T, rule=field_constraint_rule)

def crop_constraint_rule(model, i, j, s, t):
    return model.x[i, j, s, t] >= A_min * model.y[i, j, s, t]

model.crop_constraint = pyo.Constraint(model.I, model.J, model.S, model.T, rule=crop_constraint_rule)

# 引入辅助变量 z[j][t], 0-1变量, 用于表示某作物某年可以销售的量是否超过预期销售量
# 预期销售量来源于2023年数据统计, 在第一问中认为保持不变

expanded_Q = np.expand_dims(Q, axis=-1)
Q = np.tile(expanded_Q, (1, 1, 1, 7))

expanded_D = np.expand_dims(D, axis=-1)
D = np.tile(expanded_D, (1, 1, 1, 7))

model.z = pyo.Var(model.J, model.T, domain=pyo.Binary)

def sales_rule(model, i, j, s, t):
    sales = 0
    if t==2024:
        sales = sum((gov_acq_rate + pes_sale_rate) * (1 - alpha * y_2023[i - 1, j - 1, s - 1]) * Q[i - 1, j - 1, s - 1, t - 2024] * model.x[i, j, s, t] for i in model.I for s in model.S)
    else:
        sales = sum((gov_acq_rate + pes_sale_rate) * (1 - alpha * model.y[i, j, s, t]) * Q[i - 1, j - 1, s - 1, t - 2024] * model.x[i, j, s, t] for i in model.I for s in model.S)
    return sales

def cost_rule(model, i, j, s, t):
    return sum(D[i - 1, j - 1, s - 1, t - 2024] * model.x[i, j, s, t] for i in model.I for s in model.S)

model.sales = pyo.Expression(model.I, model.J, model.S, model.T, rule=sales_rule)
model.cost = pyo.Expression(model.I, model.J, model.S, model.T, rule=cost_rule)

def constraint_z_rule_1(model, i, j, s, t):
    if j in model.J and t in model.T:
        return model.sales[i, j, s, t] - sales_expectation[j] >= -M * (1 - model.z[j, t])
    return pyo.Constraint.Skip

def constraint_z_rule_2(model, i, j, s, t):
    if j in model.J and t in model.T:
        return model.sales[i, j, s, t] - sales_expectation[j] <= M * model.z[j, t]
    return pyo.Constraint.Skip

model.constraint_z_1 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=constraint_z_rule_1)
model.constraint_z_2 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=constraint_z_rule_2)


# 目标函数：总利润 = 总收入 - 总成本
# 这里涉及两种方案，分别对应第一问的两种要求

# 第一种情况：超过2023年的部分滞销。

# 多余农作物降价的比例

rate = 0

def objective_rule(model):
    return sum((1 - model.z[j, t]) * (P[j - 1] * model.sales[i, j, s, t])
    + model.z[j, t] * (P[j - 1] * (sales_expectation[j] + rate * (model.sales[i, j, s, t] - sales_expectation[j])))
    - model.cost[i, j, s, t] for i in model.I for j in model.J for s in model.S for t in model.T)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)


solver = SolverFactory('gurobi', solver_io='python')  

# 传递Gurobi参数
solver.options['NonConvex'] = 2  # 处理非凸非线性问题
solver.options['OutputFlag'] = 0  # 关闭日志输出
solver.options['LogToConsole'] = 0  # 关闭控制台日志输出

results = solver.solve(model, tee=False)

# 输出求解状态信息
# print(f"Solver Status: {results.solver.status}")
# print(f"Termination Condition: {results.solver.termination_condition}")

# 创建一个空的 DataFrame，用于存储结果
df_result = pd.DataFrame(columns=['i', 'j', 's', 't', 'x[i,j,s,t]'])

if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
    print(f"Optimal solution found: x = {model.x.value}, y = {model.y.value}, Objective = {model.obj()}")

    # 将结果添加到 DataFrame 中
    for i in model.I:
        for j in model.J:
            for s in model.S:
                for t in model.T:
                    df_result = df_result.append({'i': i, 'j': j, 's': s, 't': t, 'x[i,j,s,t]': model.x[i, j, s, t].value}, ignore_index=True)
else:
    print("No optimal solution found.")

# 将 DataFrame 输出到 Excel 文件
df_result.to_excel("solution_results.xlsx", index=False)
print("Results have been exported to solution_results.xlsx.")

# print(results)

print("Optimal objective value:", pyo.value(model.obj))