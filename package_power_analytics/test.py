import package_power_analytics.analytic as an
import pandas as pd
# import main
import sqlite3

xls = pd.ExcelFile('Тестовые данные2.xlsx')
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
# print(d_tariff.power_analyzer_day()['Максимум активной мощности в часы утреннего максимума энергосистемы , кВт'])
# # df_new = df1.set_index('Дата')

dif_tariff = an.DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
print(dif_tariff.calculation())
# print(dif_tariff.dd_power_analyzer_month())
# print(dif_tariff.dd_power_analyzer_day().iloc[:,[1,2]])
