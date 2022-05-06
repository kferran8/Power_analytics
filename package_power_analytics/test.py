import package_power_analytics.analytic as an
import pandas as pd

df = pd.read_excel('d:\Шаблон ввода исходных данных (Мул).xlsx', sheet_name='Получасовая статистика')
print(df.iloc[:,[1]])

x=df.iloc[:,[1]]

res = an.describe_statistics(x)
print(res)

