import package_power_analytics.analytic as an
import pandas as pd
# import main
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
# power_coefficients = an.PowerGraphCoefficients(df=df_power_statistics)

# df_mean = power_coefficients.calculation_mean_power_of_month()
# df_square = power_coefficients.calculation_square_power_of_month()

# print(power_coefficients.df)

df_ = df1.iloc[:,0]-pd.Timedelta(seconds=1)
df__ = df1.iloc[:,[1,2]]
merge = pd.concat([df_, df__], axis=1)

print(merge)

