import read_data, random, math
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# 阈值：落在0的epsilon邻域则认为近似是0
epsilon = 0.01

# 阈值：每块地种植的最小面积
A_min = 0.1

# 减产率
alpha = 0.25

# 政府收购比例，农民自销比例
gov_acq_rate = 0.6
pes_sale_rate = 0.3

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

# print(P)

'''
    以下注释掉的代码为调试时使用
'''

'''
def cropTypeJudge(crop) :
    crop_row = crop_data[crop_data['作物名称'] == crop]
    if not crop_row.empty:
        actual_crop_type = crop_row['作物类型'].values[0]
        return actual_crop_type
    else:
        print(f"作物 '{crop}' 不存在于数据中。")
        return False

def fieldTypeJudge(field) :
    field_row = field_data[field_data['地块名称'] == field]
    if not field_row.empty:
        actual_field_type = field_row['地块类型'].values[0]
        return actual_field_type
    else:
        print(f"地块 '{field}' 不存在于数据中。")
        return False
    
def fieldArea(field) :
    field_row = field_data[field_data['地块名称'] == field]
    if not field_row.empty:
        actual_field_area = field_row['地块面积/亩'].values[0]
        return actual_field_area
    else:
        print(f"地块 '{field}' 不存在于数据中。")
        return False
'''

# print(field_name, crop_name, field_type, crop_type)
# print(field_area)
# print(fieldArea('A341'))

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

# 引入减产惩罚机制，构建罚因子矩阵y[i, j, s]

def generate_matrix_y():
    y = np.zeros((54, 41, 2), dtype=int)
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
    return y
                
y = generate_matrix_y()          
# print(y)

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
# print(Q)

def update_Q(Q_prev):
    Q = np.zeros((54, 41, 2), dtype=float)
    for i in range(54):
        for j in range(41):
            for s in range(2):
                Q[i, j, s] = Q[i, j, s] * (1 - y[i, j, s] * alpha)
    return Q

# 生成价格矩阵，在第一问的情境下，我们认为价格趋近平稳，因此价格取定范围内的随机值
def generate_matrix_P():
    P = np.zeros((54, 41, 2), dtype=float)
    
    for index, row in stat_2023.iterrows():
        i_range = field_map[row['地块类型']]
        j = int(row['作物编号']) - 1
        s = 1 if row['种植季次'] in ['单季', '第一季'] else 2
        s -= 1
        for i in i_range:
            P[i, j, s] += row['Max']  
    return P

P = generate_matrix_P()
# print(P)

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
# print(D)
'''
  预期销售量：根据题目要求，我们认为每年的预期销售量相对于2023年保持平稳。
  在第一问中，根据相关现行政策，并统计2023年情况，我们认为预期销售量是实际产量的90%。
  其中政府收购60%，农民自销30%，剩余10%农民自种自吃或者计入损耗。
  因此首先统计2023年产量的90%作为未来预期销售量。
'''

def generate_matrix_E():
    return generate_matrix_Q_2023() * (gov_acq_rate + pes_sale_rate)    

# 定义非线性规划模型
model = pyo.ConcreteModel()

# 起始迭代年份
t = 2024
# 输出结果
res = np.zeros((54, 41, 2, 7))

# 定义索引集
model.I1 = pyo.RangeSet(1, 26)  # 平旱地、梯田和山坡地编号
model.I2 = pyo.RangeSet(27, 34)  # 水浇地编号
model.I3 = pyo.RangeSet(35, 50)  # 普通大棚编号
model.I4 = pyo.RangeSet(51, 54)  # 智慧大棚编号

model.J1 = pyo.RangeSet(16, 41)  # 作物编号 16 到 41 水稻和其它非粮食类作物
model.J2 = pyo.RangeSet(1, 15)   # 作物编号 1 到 15 粮食类作物（除水稻）
model.J3 = pyo.Set(initialize=[35, 36, 37])   # 大白菜、白萝卜和红萝卜的编号
model.J4 = pyo.RangeSet(1, 34)   # 作物编号 1 到 34 除水浇地和普通大棚第一季
model.J5 = pyo.Set(initialize=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 35, 36, 37, 38, 39, 40, 41])

model.J_soybeans_corn = pyo.RangeSet(1, 5)  # 粮食（豆类）编号
model.J_soybeans_vege = pyo.RangeSet(17, 19)    #蔬菜（豆类）编号
model.S = pyo.RangeSet(1, 2)  # 季节数

model.I = model.I1 | model.I2 | model.I3 | model.I4
model.J = model.J2 | model.J1

# 定义变量
model.x = pyo.Var(model.I, model.J, model.S, domain=pyo.NonNegativeReals)

# 约束 (2) 面积
def constraint_2(model, i, s):
    return sum(model.x[i, j, s] for j in model.J) <= field_area[i - 1]

model.Constraint_2 = pyo.Constraint(model.I, model.S, rule=constraint_2)

# 约束 (3) 平旱地、梯田和山坡地每年适宜单季种植粮食类作物（水稻除外）
def constraint_3a(model, i, j, s):
    if i in model.I1 and j in model.J1:
        return model.x[i, j, s] == 0
    return pyo.Constraint.Skip

def constraint_3b(model, i, j):
    if i in model.I1 and j in model.J2:
        return model.x[i, j, 2] == 0
    return pyo.Constraint.Skip

model.Constraint_3a = pyo.Constraint(model.I1, model.J1, model.S, rule=constraint_3a)
model.Constraint_3b = pyo.Constraint(model.I1, model.J2, rule=constraint_3b)

# 约束 (4) 水浇地每年可以单季种植水稻或两季种植蔬菜作物
def constraint_4(model, i):
    if i in model.I2:
        return model.x[i, 16, 2] == 0
    return pyo.Constraint.Skip

model.Constraint_4 = pyo.Constraint(model.I2, rule=constraint_4)

# 约束 (5) 水浇地的种植限制
def constraint_5a(model, i, j):
    if i in model.I2 and j in model.J4:
        return model.x[i, j, 2] == 0
    return pyo.Constraint.Skip

def constraint_5b(model, i, j):
    if i in model.I2 and j in model.J5:
        return model.x[i, j, 1] == 0
    return pyo.Constraint.Skip

model.Constraint_5a = pyo.Constraint(model.I2, model.J4, rule=constraint_5a)
model.Constraint_5b = pyo.Constraint(model.I2, model.J5, rule=constraint_5b)

# 约束 (6) 普通大棚每年种植两季作物，第一季可种植多种蔬菜（大白菜、白萝卜和红萝卜除外），第二季只能种植食用菌。
def constraint_6a(model, i, j):
    if i in model.I3 and j in model.J5:
        return model.x[i, j, 1] == 0
    return pyo.Constraint.Skip

def constraint_6b(model, i, j):
    if i in model.I3 and j in range(1, 38):
        return model.x[i, j, 2] == 0
    return pyo.Constraint.Skip

model.Constraint_6a = pyo.Constraint(model.I3, model.J5, rule=constraint_6a)
model.Constraint_6b = pyo.Constraint(model.I3, pyo.RangeSet(1, 37), rule=constraint_6b)

# 约束 (7) 智慧大棚每年都可种植两季蔬菜（大白菜、白萝卜和红萝卜除外）
def constraint_7(model, i, j, s):
    if i in model.I4 and j in model.J5:
        return model.x[i, j, s] == 0
    return pyo.Constraint.Skip

model.Constraint_7 = pyo.Constraint(model.I4, model.J5, model.S, rule=constraint_7)

# 约束 (8) 轮作约束，每个地块在三年内至少要种植一次豆类作物，并且水浇地第二季只能种植特定的三种作物。
def constraint_8a(model, i, j, s):
    if i in model.I1 and j in model.J_soybeans_corn and s==1:
        if t==2024:
            return data_2023[i - 1, j - 1, s - 1] + model.x[i, j, s] >= epsilon
        elif t==2025:
            return data_2023[i - 1, j - 1, s - 1] + res[i - 1, j - 1, s - 1, 0] + model.x[i, j, s] >= epsilon
        else:
            return res[i, j, s, t - 2024 - 2] + res[i, j, s, t - 1 - 2024] + model.x[i, j, s] >= epsilon
    return pyo.Constraint.Skip

model.Constraint_8a = pyo.Constraint(model.I1, model.J1, model.S, rule=constraint_8a)

def constraint_8b(model, i, j, s):
    if i in model.I2 and j in model.J2 and s == 1:
        if t==2024:
            return data_2023[i - 1, j - 1, s - 1] + model.x[i, j, s] >= epsilon
        elif t==2025:
            return data_2023[i - 1, j - 1, s - 1] + res[i - 1, j - 1, s - 1, 0] + model.x[i, j, s] >= epsilon
        else:
            return res[i, j, s, t - 2024 - 2] + res[i, j, s, t - 1 - 2024] + model.x[i, j, s] >= epsilon
    return pyo.Constraint.Skip

model.Constraint_8b = pyo.Constraint(model.I2, model.J2, model.S, rule=constraint_8b)

def constraint_8c(model, i, j, s):
    if i in model.I3 and j in model.J2:
        if t==2024:
            return data_2023[i - 1, j - 1, s - 1] + model.x[i, j, s] >= epsilon
        elif t==2025:
            return data_2023[i - 1, j - 1, s - 1] + res[i - 1, j - 1, s - 1, 0] + model.x[i, j, s] >= epsilon
        else:
            return res[i, j, s, t - 2024 - 2] + res[i, j, s, t - 1 - 2024] + model.x[i, j, s] >= epsilon
    return pyo.Constraint.Skip

model.Constraint_8c = pyo.Constraint(model.I3, model.J2, model.S, rule=constraint_8c)
'''
# 约束 (9) 种植分散性约束

'''
def dsp_2023():
    X_jst = lambda j,s: sum(data_2023[i - 1, j - 1, s] for i in model.I)
    s1 = [X_jst(j, 0) for j in model.J]
    s2 = [X_jst(j, 1) for j in model.J]
    df_xjst = pd.DataFrame({'s1':s1, 's2':s2})
    df_xjst['Max'] = df_xjst[['s1', 's2']].max(axis=1)
    df_Xjst = df_xjst['Max']
    
    distance = lambda p,q : abs(p - q)
    
    # 计算分散度指数 C_jst
    fd_dict = field_to_dict(field_keys)
    fd_dict = {value: key for key,value in fd_dict.items()}
    tp_dict = {1: range(1, 7), 2: range(7, 21), 3: range(21, 27), 4: range(27, 35), 5: range(35, 51), 6: range(51, 55)}
    x_jst = lambda p,j,s: sum(data_2023[i - 1, j - 1, s - 1] for i in tp_dict[p])
    dispersion_index = lambda j,s: sum(
        distance(p, q) * (x_jst(p, j, s) / df_Xjst[j - 1]) * (x_jst(q, j, s) / df_Xjst[j - 1])
        for p in range(1, 7) for q in range(1, 7)
    )
    
    C_th = pd.DataFrame({'s1':[dispersion_index(j, 1) for j in model.J], 's2':[dispersion_index(j, 2) for j in model.J]})
    C_th['Max'] = C_th.max().max()
    C_th = C_th['Max']
    
    return C_th

# print(dsp_2023())
C_threshold = dsp_2023()

'''
def dpx_expr(model, j, s, t):
    distance = lambda p,q : abs(p - q)
    
    # 计算分散度指数 C_jst
    fd_dict = field_to_dict(field_keys)
    fd_dict = {value: key for key,value in fd_dict.items()}
    tp_dict = {1: range(1, 7), 2: range(7, 21), 3: range(21, 27), 4: range(27, 35), 5: range(35, 51), 6: range(51, 55)}
    x_jst = lambda p, j, s: sum(model.x[i, j, s, t] for i in tp_dict[p])
    X_jst = lambda j, s, t: sum(model.x[i, j, s, t] for i in model.I)
    dispersion_index = sum(
        distance(p, q) * (x_jst(p, j, s) / (X_jst(j, s, t) + epsilon)) * (x_jst(q, j, s) / (X_jst(j, s, t) + epsilon))
        for p in range(1, 7) for q in range(1, 7)
    )
    
    return dispersion_index

model.dpx_jst = pyo.Expression(model.J,model.S,model.T, rule=dpx_expr)
'''

'''
def product_p_jst_expr(model, p, j, s, t):
    tp_dict = {1: range(1, 7), 2: range(7, 21), 3: range(21, 27), 4: range(27, 35), 5: range(35, 51), 6: range(51, 55)}
    return sum(model.x[i, j, s, t] for i in tp_dict[p])




model.product_p_jst = pyo.Expression(model.P, model.J,model.S,model.T, rule=product_p_jst_expr)
'''
'''
def linear_approx_dpx_expr(model, j, s, t):
    # 初始化近似的变量和约束
    epsilon = 1e-6  # 很小的常数，避免除以零
    tp_dict = {1: range(1, 7), 2: range(7, 21), 3: range(21, 27), 4: range(27, 35), 5: range(35, 51), 6: range(51, 55)}
    
    # 新的变量：w[p,q]代表x_jst(p,j,s) * x_jst(q,j,s)
    # 添加乘积变量约束
    for p in range(1, 7):
        for q in range(1, 7):
            # 新的乘积变量
            model.w[p, q, j, s, t] = pyo.Var(within=pyo.NonNegativeReals)

            # 线性化乘积
            model.Constraint.add(model.w[p, q, j, s, t] <= x_jst(p, j, s))
            model.Constraint.add(model.w[p, q, j, s, t] <= x_jst(q, j, s))
            model.Constraint.add(model.w[p, q, j, s, t] >= x_jst(p, j, s) + x_jst(q, j, s) - 1)

    # 重新计算线性化的 dispersion_index
    dispersion_index = sum(
        distance(p, q) * (model.w[p, q, j, s, t] / (X_jst(j, s, t) + epsilon))
        for p in range(1, 7) for q in range(1, 7)
    )

    return dispersion_index

model.dpx_jst = pyo.Expression(model.J, model.S, model.T, rule=linear_approx_dpx_expr)

def constraint_9(model,i, j, s, t):
    if i in model.I1 and j in model.J and s==2:
        return model.dpx_jst[j, 1, t] <= C_threshold[j - 1]
    return model.dpx_jst[j, s, t] <= C_threshold[j - 1]

model.Constraint_9 = pyo.Constraint(model.I, model.J, model.S, model.T, rule=constraint_9)
'''

# 约束 (10) 种植面积限制，每地块的种植面积不能少于阈值

def constraint_10(model, i, j, s):
    return sum(model.x[i, j, s] for j in model.J) >= A_min

model.Constraint_10 = pyo.Constraint(model.I, model.J, model.S, rule=constraint_10)

# 目标函数：总利润 = 总收入 - 总成本
# 这里涉及两种方案，分别对应第一问的两种要求

# 第一种情况：超过2023年的部分滞销。

def objective_rule(model):
    return sum( (gov_acq_rate + pes_sale_rate) * Q[i - 1, j - 1, s - 1] * 
               model.x[i, j, s] * P[i - 1, j - 1, s - 1] - 
               D[i - 1, j - 1, s - 1] * model.x[i, j, s]
               for i in model.I for j in model.J for s in model.S)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

# 创建一个求解器
solver = SolverFactory('gurobi', solver_io='python')
# solver.options['NonConvex'] = 2

# 运行优化器并输出结果
result_data = {}  # 存储每年优化结果的字典

while t <= 2030:
    Q = update_Q(Q)
    result = solver.solve(model, tee=True)
    print(f"Year {t}: Solver Status: {result.solver.status}, Termination Condition: {result.solver.termination_condition}")
    
    # 保存每个地块、每种作物和季节的种植面积结果
    for i in range(0, 54):
        for j in range(0, 41):
            for s in [0, 1]:
                res[i, j, s] = model.x[i + 1, j + 1, s + 1]
    y = update_matrix_y(t)
    t += 1
    
print(res)
