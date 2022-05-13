import package_power_analytics.analytic as an
import pandas as pd
import main
import sqlite3


xls = pd.ExcelFile('Исходные данные Водоканал.xlsx')
df_initial_data = xls.parse('Исх данные')
df_declared = xls.parse('Заявл мощность')
df_power_statistics = xls.parse('Получасовая статистика')

df1 = df_power_statistics
df3 = df_declared
#
# Базовый курс доллара США
kb = float(df_initial_data.iloc[2, 1])
# Текущий курс доллара США
kt = float(df_initial_data.iloc[3, 1])
# Основная плата – за мощность (на 1 месяц)
ab = float(df_initial_data.iloc[4, 1])
# Дополнительная плата – за энергию
bb = float(df_initial_data.iloc[5, 1])

#
#
# d_tariff = an.DTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
# d_tariff_declared = an.DTariffDeclared(df=df1, ab=ab, bb=bb, kt=kt, kb=kb, declared=df3)
# d_tariff_declared.calculation()
# d_tariff_decl_for_table = d_tariff_declared.df_pay_energy_month.reset_index()
power_coefficients = an.PowerGraphCoefficients(df=df_power_statistics)
# df_mean = power_coefficients.calculation_mean_power_of_month()
df_square = power_coefficients.calculation_square_power_of_month()
print(df_square)





# conn = sqlite3.connect('data.db')
# c = conn.cursor()
#
# def create_user_table():
#     c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT, password TEXT)')
#
# def add_user_data(username, password):
#     c.execute('INSERT INTO usertable(username, password) VALUES (?, ?)', (username, password))
#
# def login_user(username, password):
#     c.execute('SELECT * FROM usertable WHERE username =? and password=?', (username, password))
#     data=c.fetchall()
#     return data
#
# def view_all_user():
#     c.execute('SELECT * FROM usertable')
#     df = pd.read_sql("SELECT * FROM usertable ", conn)
#     return df
#
# create_user_table()
#
# for i in range(10,20,1):
#     print(i)
#     add_user_data(username=i, password=1111)
#
# df = pd.read_sql("SELECT * FROM usertable ", conn)
# print(df)
#
# def check_user(username, password):
#     c.execute('SELECT * FROM usertable WHERE username =? and password=?', (username, password))
#     data=c.fetchall()
#     if data is None:
#         return 0
#     else:
#         return 1
