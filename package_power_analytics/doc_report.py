from docxtpl import DocxTemplate
import sys
import power_analytics.table_contents as tbl
import power_analytics.figure as fgr
import power_analytics.analytic as analytic
import pandas as pd
import datetime

def create_report(df_input_statistics, df_initial_data, df_declared_power):

    # Создаем шаблон
    try:
        template = DocxTemplate("Tamplate/Tamplate.docx")
    except Exception:
        print('Отсутсвует шаблон заполнения отчета')
        sys.exit()
    # Создаем контейнер хранения данных для отчета
    context = {}
    # Дата на титульном листе
    context['year_now'] = datetime.datetime.now().year

    # __________Исходные данные о тарифах и названии организации ________
    df1 = df_input_statistics
    df3 = df_declared_power
    # Базовый курс доллара США
    kb = df_initial_data.iloc[2, 1]
    # Текущий курс доллара США
    kt = df_initial_data.iloc[3, 1]
    # Основная плата – за мощность (на 1 месяц)
    ab = df_initial_data.iloc[4, 1]
    # Дополнительная плата – за энергию
    bb = df_initial_data.iloc[5, 1]
    print(df1)
    print(df3)
    company_name = df_initial_data.iloc[0, 1]
    point_of_measurement = df_initial_data.iloc[1, 1]
    # Дата декларации тарифов
    date_of_declaration = df_initial_data.iloc[6, 1]
    date_of_exchange = df_initial_data.iloc[7, 1].date()
    base_tariff = df_initial_data.iloc[8, 1]

    # __________Запускаем аналитику и создаем экземпляры класса ________
    power_coefficients = analytic.PowerGraphCoefficients(df=df1)
    power_limits = analytic.PowerLimits(df=df1)
    d_tariff = analytic.DTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
    d_tariff_declared = analytic.DTariffDeclared(df=df1, ab=ab, bb=bb, kt=kt, kb=kb, declared=df3)
    dif_tariff = analytic.DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
    dif_tariff_reg = analytic.DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)

    # ____________ ВВЕДЕНИЕ_____________________#

    context['company_name'] = company_name

    # ____________ ГЛАВА 1_____________________#
    # ____________ раздел 1.1_____________________#

    context['point_of_measurement'] = point_of_measurement
    count_day = power_coefficients.count_day

    context['Кол_дней'] = count_day
    context['Макс_год'] = power_coefficients.df_rename.index.year.max()
    context['Мин_год'] = power_coefficients.df_rename.index.year.min()

    # Таблица 1.1 Фрагмент статистических наблюдения электрической нагрузки
    df1 = power_coefficients.df
    table_contents_1_1 = tbl.table_dynamic(data=df1[:35])
    context['table_contents_1_1'] = table_contents_1_1

    # Рисунок 1.1 Исходные наблюдения активной мощности
    act_power = power_coefficients.df_rename.iloc[:, [0]]
    image11 = fgr.my_subplots(template=template, data=act_power, name_fig = 'График активной мощности')
    context['Исх_граф_акт'] = image11

    # Рисунок 1.2 Исходные наблюдения реактивной мощности
    react_power = power_coefficients.df_rename.iloc[:, [1]]
    image12 = fgr.my_subplots(template=template, data=react_power, name_fig = 'График реактивной мощности')
    context['Исх_граф_реакт'] = image12

    # Вывод средних нагрузок за исследуемый период по тексту
    df_mean = power_coefficients.calculation_mean_power_of_month().set_index('Период наблюдений')
    context['Акт_ср'] = round(df_mean.iloc[:, 0].mean(), 2)
    context['Реакт_ср'] = round(df_mean.iloc[:, 1].mean(), 2)
    context['Полн_ср'] = round(df_mean.iloc[:, 2].mean(), 2)

    # Таблица 1.2 Средняя нагрузка по месяцам
    df_mean = power_coefficients.calculation_mean_power_of_month()
    table_contents_1_2 = tbl.table_dynamic(data=df_mean)
    context['table_contents_1_2'] = table_contents_1_2

    # Рисунок 1.3 Диаграмма средней активной, реактивной и полной мощности
    image13 = fgr.my_bar(template=template, data=df_mean.set_index('Период наблюдений'),
                         name_fig = 'Средняя мощность')
    context['Средняя_мощность'] = image13

    # Таблица 1.3 Среднеквадратичная нагрузка по месяцам
    df_square = power_coefficients.calculation_square_power_of_month()

    table_contents_1_3 = tbl.table_dynamic(data=df_square)
    context['table_contents_1_3'] = table_contents_1_3
    #
    # Рисунок 1.4 Диаграмма среднекдратичной активной, реактивной и полной мощности
    image14 = fgr.my_bar(template=template, data=df_square.set_index('Период наблюдений'),
                         name_fig = 'Среднеквадратичная мощность')
    context['Среднекдваратичная_мощность'] = image14

    # Вывод средних нагрузок за исследуемый период по тексту
    context['Акт_кв'] = round(df_square.iloc[:, 1].mean(), 2)
    context['Реакт_кв'] = round(df_square.iloc[:, 2].mean(), 2)
    context['Полн_кв'] = round(df_square.iloc[:, 3].mean(), 2)

    # Таблица 1.4 Максимальная нагрузка по месяцам
    df_max = power_coefficients.calculation_max_power_of_month()

    table_contents_1_4 = tbl.table_dynamic(data=df_max)
    context['table_contents_1_4'] = table_contents_1_4

    # Рисунок 1.5 Диаграмма максимальной активной, реактивной и полной мощности
    image15 = fgr.my_bar(template=template, data=df_max.set_index('Период наблюдений'),
                         name_fig = 'Макс мощность' )
    context['Максимальная_мощность'] = image15

    # Вывод средних нагрузок за исследуемый период по тексту
    context['Акт_макс'] = round(df_max.iloc[:, 1].max(), 2)
    context['Реакт_макс'] = round(df_max.iloc[:, 2].max(), 2)
    context['Полн_макс'] = round(df_max.iloc[:, 3].max(), 2)

    # ____________ КОНЕЦ раздел 1.1_____________________#

    # ____________ раздел 1.2_____________________#
    # ______________
    # Таблица 1.5 Коэффициент максимума
    coefficient_max = power_coefficients.coefficient_max()

    table_contents_1_5 = tbl.table_dynamic(data=coefficient_max)
    context['table_contents_1_5'] = table_contents_1_5
    ylim_min = coefficient_max.iloc[:, 1].min() * 0.9
    # Рисунок 1.6 Диаграмма коэффициента максимума
    image16 = fgr.my_bar(template=template, data=coefficient_max.set_index('Период наблюдений'), colormap='ocean',
                         ylim_min=ylim_min, name_fig = 'Коэффициент максимума')
    context['Коэффициент_максимума'] = image16

    # Выводы
    context['Наиб_коэф_макс'] = round(coefficient_max.iloc[:, 1].max(), 2)
    df_temp = coefficient_max.loc[coefficient_max.iloc[:, 1] == coefficient_max.iloc[:, 1].max()]
    max_mohtn_coef = df_temp.iloc[:, 0].values[0]
    context['Мес_коэф_макс'] = max_mohtn_coef
    context['Ср_коэф_макс'] = round(coefficient_max.iloc[:, 1].mean(), 2)

    # ______________
    # Таблица 1.6 Коэффициент заполнения
    coefficient_fill = power_coefficients.coefficient_fill()
    table_contents_1_6 = tbl.table_dynamic(data=coefficient_fill)
    context['table_contents_1_6'] = table_contents_1_6

    # Рисунок 1.7 Диаграмма коэффициента заполнения
    ylim_min = coefficient_fill.iloc[:, 1].min() * 0.9
    image17 = fgr.my_bar(template=template, data=coefficient_fill.set_index('Период наблюдений'), colormap='ocean',
                         ylim_min=ylim_min, name_fig = 'Коэффициент заполнения' )
    context['Коэффициент_заполнения'] = image17


    # Выводы
    context['Наиб_коэф_запол'] = round(coefficient_fill.iloc[:, 1].max(), 2)
    df_temp = coefficient_fill.loc[coefficient_fill.iloc[:, 1] == coefficient_fill.iloc[:, 1].max()]
    max_mohtn_coef = df_temp.iloc[:, 0].values[0]
    context['Мес_коэф_запол'] = max_mohtn_coef
    context['Ср_коэф_запол'] = round(coefficient_fill.iloc[:, 1].mean(), 2)
    context['Заключ_коэф_запол'] = 'удовлетворителен' if context['Ср_коэф_запол'] > 0.5 else 'не удовлетворителен'

    # ______________
    # Таблица 1.7 Коэффициент формы
    coefficient_shape = power_coefficients.coefficient_shape()
    table_contents_1_7 = tbl.table_dynamic(data=coefficient_shape)
    context['table_contents_1_7'] = table_contents_1_7

    # Рисунок 1.8 Диаграмма коэффициента формы
    ylim_min = coefficient_shape.iloc[:, 1].min() * 0.9
    image18 = fgr.my_bar(template=template, data=coefficient_shape.set_index('Период наблюдений'),
                         name_fig='Коэффициент формы', colormap='ocean', ylim_min=ylim_min)
    context['Коэффициент_формы'] = image18

    # Выводы
    context['Наиб_коэф_формы'] = round(coefficient_shape.iloc[:, 1].max(), 2)
    df_temp = coefficient_shape.loc[coefficient_shape.iloc[:, 1] == coefficient_shape.iloc[:, 1].max()]
    max_mohtn_coef = df_temp.iloc[:, 0].values[0]
    context['Мес_коэф_формы'] = max_mohtn_coef
    context['Ср_коэф_формы'] = round(coefficient_shape.iloc[:, 1].mean(), 2)
    context['Заключ_коэф_формы'] = 'не рациональной' if context['Ср_коэф_запол'] > 1.5 else 'рациональной'

    # ______________
    # Таблица 1.8 Коэффициент мощности косинус фи
    coefficient_fi = power_coefficients.coefficient_fi()
    table_contents_1_8 = tbl.table_dynamic(data=coefficient_fi)
    context['table_contents_1_8'] = table_contents_1_8

    # Рисунок 1.9 Диаграмма коэффициента мощности
    ylim_min = coefficient_fi.iloc[:, 1].min() * 0.9
    image19 = fgr.my_bar(template=template, data=coefficient_fi.set_index('Период наблюдений'),
                         name_fig = 'Коэффициент мощности', colormap='ocean', ylim_min=ylim_min)
    context['Диагр_коэф_акт_мощ'] = image19

    # Выводы
    context['Наиб_коэф_акт_мощ'] = round(coefficient_fi.iloc[:, 1].max(), 2)
    context['Наим_коэф_акт_мощ'] = round(coefficient_fi.iloc[:, 1].min(), 2)
    context['Ср_коэф_акт_мощ'] = round(coefficient_fi.iloc[:, 1].mean(), 2)
    context['Заключ_коэф_акт_мощ'] = 'об эффективной' if context['Ср_коэф_акт_мощ'] > 0.85 else 'о не эффективной'

    # Рисунок 1.10 График коэффициента мощности
    df_coef_fi_day = power_coefficients.df_coef_fi_day
    image110 = fgr.my_subplots(template=template, data=df_coef_fi_day, name_fig='График коэффициента мощности')
    context['График_коэф_акт_мощ'] = image110

    # Записываем результаты в эксель файл
    power_coefficients.write_to_exel()


    # ____________ ГЛАВА 2_____________________#
    context['Дата_декларации'] = date_of_declaration
    context['БазКурс'] = kb
    context['ОснСтавка'] = ab
    context['ДопСтавка'] = bb
    context['ТекКурс'] = kt
    context['ДатаКурсаДолл'] = date_of_exchange
    context['ИндОснСтавка'] = d_tariff.tariff_power
    context['ИндДопСтавка'] = d_tariff.tariff_energy


    # ____________ ГЛАВА 3_____________________#
    #Таблица 3.1 вывод результатов мощностей в часы максимума энергосистемы
    df_full_limit = power_limits.power_limits()
    df_only_max_period= power_limits.df_only_max_period
    df_only_max_period = df_only_max_period.iloc[:,[0,4,5,6,7,8,9,10]]
    table_contents_2_1 = tbl.table_dynamic(data=df_only_max_period)
    context['table_contents_2_1'] = table_contents_2_1
    df_full_limit = df_full_limit.rename(columns={'Month':'Месяц'})

    # Рисунок 3.1 вывод графиков сравнения мощностей максимальной и предельной
    drop_list = ['Год', 'Месяц', 'Час', 'Часы суток',
       'Минимальная активная мощность, кВт',
       'Среднеквадратическое отклонение активной мощности, кВт',
       'Количество значений']

    df_full_limit = df_full_limit.drop(drop_list, axis = 1)
    image21 = fgr.my_plot_limit(template=template, data=df_full_limit, name_fig = 'Диаграмма лимитов')
    context['Диаг_лимит'] = image21

    # Таблица 3.2 сводные лимиты по месяцам
    df_max_month_value = power_limits.df_max_month_value.reset_index()
    df_max_month_value = df_max_month_value.iloc[:,[1,2,3]]
    table_contents_2_2 = tbl.table_dynamic(data=df_max_month_value)
    context['table_contents_2_2'] = table_contents_2_2

    df_max_month_value = power_limits.df_max_month_value


    image22 = fgr.my_bar(template=template, data=df_max_month_value.set_index('Период наблюдений'),
                         name_fig = 'Диаграмма лимитов мощности по месяцам',  ylim_min=ylim_min)
    context['Диаг_лимит_мес'] = image22


    # ____________ РАЗДЕЛ 4.1_Д-тариф с заявленной мощностью____________________#

    # Таблица 4.2 Расчет платы за электроэнергию при оплате по двухставочному тарифу с заявленной мощностью
    d_tariff_declared.calculation()
    d_tariff_decl_for_table = d_tariff_declared.df_pay_energy_month.reset_index()
    table_contents_4_2 = tbl.table_dynamic(data=d_tariff_decl_for_table)
    context['table_contents_4_2'] = table_contents_4_2
    print(d_tariff_decl_for_table)
    # Суммарная плата за мощность и энергию
    sum_pay_power_and_energy = d_tariff_declared.sum_pay_power_and_energy
    context['DecTSumPay'] = sum_pay_power_and_energy
    #Суммарная плата за мощность
    sum_pay_power = d_tariff_declared.sum_pay_power
    context['DecTSumPayPower'] = sum_pay_power
    #Суммарная плата за ЭЭ
    sum_pay_energy = d_tariff_declared.sum_pay_energy
    context['DecTSumPayEn'] = sum_pay_energy
    # процент оплаты за мощность в суммарной оплате
    dec_per = round(sum_pay_power / sum_pay_power_and_energy * 100, 1)
    context['DecPer'] = dec_per

    # Построение диаграммы 4.1
    dt_bar = d_tariff_declared.df_pay_energy_month.iloc[:, [4, 7, 8]]
    image41 = fgr.my_bar(template=template, data=dt_bar,
                         name_fig='Д-тариф заявл сравнение оплаты', ylim_min=ylim_min)
    context['ДиагрОплатыДТарифЗаявл'] = image41


    #Рисунок 4.4 Круговая диаграмма по тарифным зонам
    list_pay_dec = [sum_pay_energy, sum_pay_power]
    list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']
    image42 = fgr.my_pie(template=template, list_data=list_pay_dec, labels=list_labels, name_fig='КругЗаявл')
    context['КругЗаявл'] = image42
    d_tariff_declared.write_to_exel()
    power_limits.write_to_exel()


    # ____________ РАЗДЕЛ 4.2_Д-тариф с фактической мощностью____________________#


    # Таблица 4.3 Максимумы 30-и минутной активной мощности в расчетном периоде
    power_analyzer_month = d_tariff.power_analyzer_month().reset_index()
    table_contents_4_3 = tbl.table_dynamic(data=power_analyzer_month)
    context['table_contents_4_3'] = table_contents_4_3
    print(power_analyzer_month)
    #Количество месяцев исследования
    count_month = power_analyzer_month['Индикатор'].count()
    #Количество месяцев где наибольший максимум вышел за границы максимума энергосистемы
    count_month_true = power_analyzer_month['Индикатор'].sum()
    context['count_month'] = count_month
    context['count_month_true'] = count_month_true
    power = power_analyzer_month.set_index('Период наблюдений')
    power = power.iloc[:,[0,1,2]]
    image43 = fgr.my_bar(template=template, data=power,
                         name_fig='Диаграмма лимитов мощности по месяцам', ylim_min=ylim_min)
    context['СравнМаксимумов'] = image43

    #Приложение Таблица П1 Сравнение мощностей по суткам
    power_analyzer_day = d_tariff.power_analyzer_day().reset_index()
    table_contents_P1 = tbl.table_dynamic(data=power_analyzer_day)
    context['table_contents_P1'] = table_contents_P1

    #Количество суток исследования
    count_day = power_analyzer_day['Индикатор'].count()
    # Количество суток где наибольший максимум вышел за границы максимума энергосистемы
    count_day_true = power_analyzer_day['Индикатор'].sum()
    context['count_day'] = count_day
    context['count_day_true'] = count_day_true
    val = round(count_day_true /  count_day * 100, )
    if val > 50:
        conclusion1 = f'регулирование 30-и минутных максимумов нагрузок можно считать эффективным, ' \
              f'поскольку преимущественно  ({val} % случаев) ' \
              f'максимальная нагрузка выходит за границы максимума энергосистемы'
    else:
        conclusion1 = f'регулирование 30-и минутных максимумов максимумов нагрузок является низкоэффективным, ' \
              f'поскольку преимущественно  ({val} % случаев) ' \
              f'максимальная нагрузка находится в границах максимума энергосистемы'
    context['conclusion1'] = conclusion1


    # Таблица 4.4 Анализ электропотребления за расчетный период исследования
    energy_analyzer = d_tariff.energy_analyzer_month().reset_index()
    table_contents_4_4 = tbl.table_dynamic(data=energy_analyzer)
    context['table_contents_4_4'] = table_contents_4_4
    print(energy_analyzer)
    #Суммарный расход ЭЭ
    sum_energy = d_tariff.sum_energy
    context['SumEnergy'] = sum_energy


    # Таблица 4.5 Расчет платы за электроэнергию при оплате по двухставочному тарифу
    d_tariff.calculation()
    pay_month = d_tariff.df_pay_energy_month.reset_index()
    table_contents_4_5 = tbl.table_dynamic(data=pay_month)
    context['table_contents_4_5'] = table_contents_4_5
    print(pay_month)
    #Суммарная плата за ЭЭ
    sum_pay_power = d_tariff.sum_pay_power
    context['DTarSumPayPower'] = sum_pay_power
    sum_pay_energy = d_tariff.sum_pay_energy
    context['DTarSumPayEn'] = sum_pay_energy
    sum_pay_power_and_energy = d_tariff.sum_pay_power_and_energy
    context['DTarSumPay'] = sum_pay_power_and_energy
    dt_per = round(sum_pay_power / sum_pay_power_and_energy * 100, 1)
    context['DTPer'] = dt_per

    dt_bar = d_tariff.df_pay_energy_month.iloc[:, [4, 7, 8]]
    image44 = fgr.my_bar(template=template, data=dt_bar,
                         name_fig='Д-тариф сравнение стоимости оплаты', ylim_min=ylim_min)
    context['ДиагрОплатыДТариф'] = image44

    list_pay_fact = [sum_pay_energy, sum_pay_power]
    list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']
    image45 = fgr.my_pie(template=template, list_data=list_pay_fact, labels=list_labels, name_fig='КругФакт')
    context['КругФакт'] = image45
    d_tariff.write_to_exel()


    # ____________ РАЗДЕЛ 4.3 ДД-тариф_____________________#

    # Таблица 4.5 тарифные коэффициенты
    tariff_coefficients = dif_tariff.calculation_tariff_coefficients().reset_index()
    table_contents_4_6 = tbl.table_dynamic(data=tariff_coefficients)
    context['table_contents_4_6'] = table_contents_4_6

    # Таблица 4.6 Расчет затрат ЭЭ по тарифным зонам
    energy_analyzer_month = dif_tariff.dd_energy_analyzer_month().reset_index()
    table_contents_4_7 = tbl.table_dynamic(data=energy_analyzer_month)
    context['table_contents_4_7'] = table_contents_4_7
    #Расход электроэнергии по зонам
    wn = dif_tariff.sum_energy_night
    context['Wn'] = wn
    wp = dif_tariff.sum_energy_peak
    context['Wp'] = wp
    wpp = dif_tariff.sum_energy_half_peak
    context['Wpp'] = wpp
    # Баланс распределения затрат ЭЭ по зонам тарифным
    wn_per = round(wn / sum_energy * 100, 1)
    wp_per = round(wp / sum_energy * 100, 1)
    wpp_per = round(wpp / sum_energy * 100, 1)
    context['WnPer'] = wn_per
    context['WpPer'] = wp_per
    context['WppPer'] = wpp_per

    #Рисунок 4.4 Круговая диаграмма по тарифным зонам
    list_pie = [dif_tariff.sum_energy_night, dif_tariff.sum_energy_peak, dif_tariff.sum_energy_half_peak]
    list_labels = ['Расход электроэнергии в ночной зоне', 'Расход электроэнергии в пиковой зоне',
                   'Расход электроэнергии в полупиковой зоне']
    image46 = fgr.my_pie(template=template, list_data=list_pie, labels=list_labels, name_fig='Круговая')
    context['Круговая'] = image46

    # Таблица 4.8 Сравнение мощностей ДД-тарифа вечернего и утреннего пиков
    dd_power_analyzer_month = dif_tariff.dd_power_analyzer_month().reset_index()
    table_contents_4_8 = tbl.table_dynamic(data=dd_power_analyzer_month)
    context['table_contents_4_8'] = table_contents_4_8
    print(dd_power_analyzer_month)
    # count_tariff
    #Проверяем количество днейе где выполняется условие оплаты
    count_temp = dd_power_analyzer_month.iloc[:,[6]]
    count_tariff = (count_temp['Выполняется условие оплаты по ДД-тарифу?'] == 'Да').sum()
    context['count_tariff'] = count_tariff

    # Таблица 4.9 - Расчет суммарной платы за электрическую энергию и мощность при
    # использовании двухставочно-дифференцированного тарифа
    dif_tariff.calculation()
    #Отбираем нужные данные для заполнения таблицы итогов
    dd_pay_energy_month = dif_tariff.df_pay_energy_month.reset_index().iloc[:,[0, 14, 15, 16, 17, 18, 19]]
    table_contents_4_9 = tbl.table_dynamic(data=dd_pay_energy_month)
    context['table_contents_4_9'] = table_contents_4_9
    print(dd_pay_energy_month)
    dd_tar_pay_power_and_energy_v1 = dif_tariff.sum_pay_power_and_energy
    context['DDTarSumPay'] = dd_tar_pay_power_and_energy_v1
    dd_tar_sum_pay_power_v1 = dif_tariff.sum_pay_power
    context['DDTarSumPayPower'] = dd_tar_sum_pay_power_v1
    dd_tar_sum_pay_energy_v1 = dif_tariff.sum_pay_energy
    context['DDTarSumPayEn'] = dd_tar_sum_pay_energy_v1
    dd_tar_per_power_v1 = round((dd_tar_sum_pay_power_v1 / dd_tar_pay_power_and_energy_v1 * 100), 1)
    context['DDTPer'] = dd_tar_per_power_v1

    # Рисунок 4.7 – Диаграмма оплаты за электрическую фактическую мощность
    dd_bar = dif_tariff.df_pay_energy_month.iloc[:,[13, 17, 18]]
    image47 = fgr.my_bar(template=template, data=dd_bar,
                         name_fig='Диаграмма ДД-тариф', ylim_min=ylim_min)
    context['ДиагрОплатыДДТариф'] = image47

    #Рисунок 4.8 – Круговая диаграмма суммарной оплаты
    list_pay_fact = [dd_tar_sum_pay_energy_v1, dd_tar_sum_pay_power_v1]
    list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']
    image48 = fgr.my_pie(template=template, list_data=list_pay_fact, labels=list_labels, name_fig='КругФактДД')
    context['КругФактДД'] = image48
    dif_tariff.write_to_exel(name_file='Расчет оплаты ДД-тарифа без регулирования')


    # ____________ РАЗДЕЛ 4.4 ДД-тариф с регулированием максимумов_____________________#
    dif_tariff_reg.calculation(type_tariff=1)
    df_tar_calc_power = dif_tariff_reg.df_pay_energy_month.reset_index().iloc[:,[0, 2, 5, 3, 4, 14]]
    table_contents_4_10 = tbl.table_dynamic(data=df_tar_calc_power)
    context['table_contents_4_10'] = table_contents_4_10
    sum_pay_power_dd = dif_tariff_reg.sum_pay_power
    context['SumPayPowDD'] = sum_pay_power_dd

    df_tar_calc_energy =  dif_tariff_reg.df_pay_energy_month.reset_index().iloc[:,[0, 2, 6, 8, 7, 15, 10, 9,
                                                                               16, 12, 11, 17, 18]]
    table_contents_4_11 = tbl.table_dynamic(data=df_tar_calc_energy)
    context['table_contents_4_11'] = table_contents_4_11
    print(df_tar_calc_energy)
    dd_pay_ener = dif_tariff_reg.sum_pay_energy
    context['SumPayEnerDD'] = dd_pay_ener
    dd_pay_ener_night = dif_tariff_reg.sum_energy_night
    context['PayWn'] = dd_pay_ener_night
    dd_pay_ener_half_peak = dif_tariff_reg.sum_energy_half_peak
    context['PayWpp'] = dd_pay_ener_half_peak
    dd_pay_ener_peak = dif_tariff_reg.sum_energy_peak
    context['PayWp'] = dd_pay_ener_peak

    # Рисунок 4.9 – Диаграмма оплаты за электрическую энергию по тарифным зонам двухставочно-дифференцированного тарифа
    dd_bar = dif_tariff_reg.df_pay_energy_month.iloc[:,[14, 15, 16]]
    image49 = fgr.my_bar(template=template, data=dd_bar,
                         name_fig='Диаграмма ДД-тариф по тарифным зонам', ylim_min=ylim_min)
    context['ДиагрОплатыДДТарифЭнергия'] = image49

    #Отбираем нужные данные для заполнения таблицы итогов
    only_dd_pay_energy_month = dif_tariff_reg.df_pay_energy_month.reset_index().iloc[:,[0, 14, 15, 16, 17, 18, 19]]
    table_contents_4_12 = tbl.table_dynamic(data=only_dd_pay_energy_month)
    context['table_contents_4_12'] = table_contents_4_12
    print(only_dd_pay_energy_month)
    dd_tar_pay_power_and_energy_v2 = dif_tariff_reg.sum_pay_power_and_energy
    context['OnlyDDTarSumPay'] = dd_tar_pay_power_and_energy_v2
    dd_tar_sum_pay_power_v2 = dif_tariff_reg.sum_pay_power
    context['OnlyDDTarSumPayPower'] = dd_tar_sum_pay_power_v2
    dd_tar_sum_pay_energy_v2 = dif_tariff_reg.sum_pay_energy
    context['OnlyDDTarSumPayEn'] = dd_tar_sum_pay_energy_v2
    dd_tar_per_power_v2 = round((dd_tar_sum_pay_power_v2 / dd_tar_pay_power_and_energy_v2 * 100), 1)
    context['OnlyDDTPer'] = dd_tar_per_power_v2

    # Рисунок 4.10 – Диаграмма оплаты за электрическую фактическую мощность
    dd_bar = dif_tariff_reg.df_pay_energy_month.iloc[:, [13, 17, 18]]
    image410 = fgr.my_bar(template=template, data=dd_bar,
                         name_fig='Диаграмма с регулир ДД-тариф', ylim_min=ylim_min)
    context['ДиагрОплатыРегулирДДТариф'] = image410

    # Рисунок 4.11 – Круговая диаграмма суммарной оплаты
    list_pay_fact = [dd_tar_sum_pay_energy_v2, dd_tar_sum_pay_power_v2]
    list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']
    image411 = fgr.my_pie(template=template, list_data=list_pay_fact, labels=list_labels, name_fig='КругФактДДсРегул')
    context['КруговаяСРегулир'] = image411

    # Заполняем таблицу приложения П1
    dd_power_analyzer_day = dif_tariff_reg.dd_power_analyzer_day().reset_index()
    table_contents_P2 = tbl.table_dynamic(data=dd_power_analyzer_day)
    context['table_contents_P2'] = table_contents_P2
    # Считаем количество дней где вечер превышает утро
    higher_day = dif_tariff_reg.dd_power_analyzer_day()['Индикатор'].sum()
    context['higher_day'] = higher_day
    # Процент превышения вечернего пика
    higher_per = round(higher_day / count_day  * 100, 1)
    context['higher_per'] = higher_per

    higher_pow = round(dif_tariff_reg.
                       dd_power_analyzer_day()['Мощность превышения вечернего максимума утренним, кВт'].mean(), 1)
    context['higher_pow'] = higher_pow

    # делаем вывод
    if higher_per > 50:
        conclusion2 = 'регулирование и контроль не превышения вечернего пика над утренним затруднительно, так как в ' \
                      'более 50 % случаях условие применения дифференцированного тарифа не соблюдается'
    else:
        conclusion2 = f'регулирование и контроль не превышения вечернего пика над утренним возможен, так как ' \
                      f'в менее 50 % случаях соблюдается условие применения дифференцированного,' \
                      f' при этом целесообразно предусмотреть регулированную мощность отключения ' \
                      f'нагрузки в среднем на {higher_pow} кВт'

    context['conclusion2'] = conclusion2
    dif_tariff_reg.write_to_exel(name_file='Расчет оплаты ДД-тарифа с регулированием')
    print(conclusion2)

    #_____________ГЛАВА 5 ОПРЕДЕЛЕНИЕ ВРЕМЕНИ ИСПОЛЬЗОВАНИЯ НАГРУЗОК_______________№
    research_hour = power_coefficients.df_rename['Час'].count()*0.5
    context['ResearchHour'] = research_hour
    research_day = research_hour / 24
    context['ResearchDay'] = research_day
    p_max = power_coefficients.df_rename['Активная мощность, кВт'].max()
    context['Pmax'] = p_max
    t_max = round(sum_energy / p_max * 365 / research_day,0)
    context['Tm'] = t_max


    #_____________ГЛАВА 6 СРАВНЕНИЕ ТАРИФОВ_______________№

    energy = energy_analyzer.set_index('Период наблюдений').iloc[:,[3]]
    declared_pay = d_tariff_decl_for_table.set_index('Период наблюдений').iloc[:,[8]]  # Суммарная оплата за ЭЭ
    d_pay = pay_month.set_index('Период наблюдений').iloc[:,[8]]  # Суммарная оплата за ЭЭ
    dd_pay = dd_pay_energy_month.set_index('Период наблюдений').iloc[:,[5]] # Суммарная оплата за ЭЭ
    dd_pay_regul = only_dd_pay_energy_month.set_index('Период наблюдений').iloc[:,[5]]  # Суммарная оплата за ЭЭ
    #Считаем среднюю стоимость 1 кВтч для заполнения таблицы
    total_pay = analytic.compare(energy=energy, declared_pay=declared_pay, d_pay=d_pay,
                           dd_pay=dd_pay, dd_pay_regul=dd_pay_regul)

    table_contents_6_1 = tbl.table_dynamic(data=total_pay.reset_index())
    context['table_contents_6_1'] = table_contents_6_1

    conclusion3 = None
    if base_tariff == 0:
        context['BaseTariff'] = 'двухставочный тариф с оплатой за заявленную мощность'
    elif base_tariff == 1:
        context['BaseTariff'] = 'двухставочный тариф с оплатой за фактическую мощность'
    elif base_tariff == 2:
        context['BaseTariff'] = 'двухставочно-дифференцированный тариф с оплатой за фактическую мощность'

    # Таблица 6.2 – Агрегированные результаты сравнения тарифов
    df_pay = pd.DataFrame({'Тариф оплаты за электроэнергию':
                               ['Д-тариф с оплатой за заявленную мощность',
                                'Д-тариф с оплатой за фактическую мощность',
                                'ДД-тариф с оплатой за фактическую мощность без изменения режима работы',
                                'ДД-тариф с оплатой за фактическую мощность при регулировании '
                                'не превышения вечернего максимума'],
                           'Суммарная оплата за мощность, руб.':
                               [d_tariff_declared.sum_pay_power, d_tariff.sum_pay_power,
                                dif_tariff.sum_pay_power, dif_tariff_reg.sum_pay_power],
                           'Суммарная оплата за электроэнергию, руб.':
                               [d_tariff_declared.sum_pay_energy, d_tariff.sum_pay_energy,
                                dif_tariff.sum_pay_energy, dif_tariff_reg.sum_pay_energy],
                           'Суммарная оплата за электроэнергию и мощность, руб.':
                           [d_tariff_declared.sum_pay_power_and_energy, d_tariff.sum_pay_power_and_energy,
                            dif_tariff.sum_pay_power_and_energy,  dif_tariff_reg.sum_pay_power_and_energy],
                           'Средний тариф за 1 кВт·ч электроэнергии, руб./ кВт·ч':
                           [d_tariff_declared.mean_tariff, d_tariff.mean_tariff, dif_tariff.mean_tariff,
                            dif_tariff_reg.mean_tariff],
                           'Индикатор наименования тарифа': [0, 1, 2, 3]
                           })

    min_pay = df_pay['Суммарная оплата за электроэнергию и мощность, руб.'].min()

    ind_opt = int(df_pay[df_pay.iloc[:, 3] == min_pay]['Индикатор наименования тарифа'].values[0])

    tar_opt = df_pay[df_pay.iloc[:, 3] == min_pay]['Тариф оплаты за электроэнергию'].values[0]
    pay_opt = float(df_pay[df_pay.iloc[:, 3] == min_pay]['Суммарная оплата за электроэнергию и мощность, руб.'].values[0])
    # Оптимальная стоимость 1 кВтч
    energy_unit_cost_opt = float(
        df_pay[df_pay.iloc[:, 3] == min_pay]['Средний тариф за 1 кВт·ч электроэнергии, руб./ кВт·ч'].values[0])

    tar_exist = df_pay[df_pay.iloc[:, 5] == base_tariff]['Тариф оплаты за электроэнергию'].values[0]
    pay_exist = float(df_pay[df_pay.iloc[:, 5] == base_tariff]['Суммарная оплата за электроэнергию и мощность, руб.'].
                      values[0])

    energy_unit_pay_exist = float(
        df_pay[df_pay.iloc[:, 5] == base_tariff]['Средний тариф за 1 кВт·ч электроэнергии, руб./ кВт·ч'].values[0])
    delta_pay = pay_exist - pay_opt

    df_pay = df_pay.sort_values(by='Суммарная оплата за электроэнергию и мощность, руб.')

    df_pay['Эффективность'] = round((pay_exist -
                                     df_pay['Суммарная оплата за электроэнергию и мощность, руб.'])/pay_exist*100, 3)

    df_pay_fill = df_pay.drop(['Индикатор наименования тарифа'], axis=1)

    table_contents_6_2 = tbl.table_dynamic(data=df_pay_fill)
    context['table_contents_6_2'] = table_contents_6_2

    # Делаем выводы по результатам расчета тарифов
    if base_tariff < ind_opt or base_tariff > ind_opt:
        conclusion3 = f'\t1. Существующий тариф оплаты за электрическую энергию ({tar_exist}) не является оптимальным. \n\t' \
                      f'2. Переход на {tar_opt} обеспечит экономический эффект в размере {delta_pay} руб. и ' \
                      f'снижение стоимости 1 кВтч с {energy_unit_pay_exist} до {energy_unit_cost_opt} руб/кВтч. \n\t'
        if base_tariff == 1:
            temp_pay = float(df_pay[df_pay.iloc[:, 5] == 3]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - pay_opt
            temp_cons = f'3. Переход на дифференцированную форму оплаты за электроэнергию даже в случае полного ' \
                        f'контроля не превышение вечернего минимума над утренним приведет к увеличению ' \
                        f'стоимости электроэнергии на {temp_pay} руб. \n\t' \
                        f'4. Переход к дифференцированной форме оплаты будет эффективен только в случае значительного ' \
                        f'изменения графика работы предприятия.'
            conclusion3 = conclusion3 + temp_cons
        if base_tariff == 2:
            temp_pay = float(df_pay[df_pay.iloc[:, 5] == 1]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - pay_opt
            temp_pay_1 = float(df_pay[df_pay.iloc[:, 5] == 1]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - \
                         float(df_pay[df_pay.iloc[:, 5] == 2]
                             ['Суммарная оплата за электроэнергию и мощность, руб.'])
            if temp_pay_1 == 0:
                temp_cons_2 = f'Переход на двухставочную форму оплаты не приведет к изменению стоимости ' \
                              f'электроэнергии (разница тарифов {temp_pay_1} руб.) без регулирования суммарного ' \
                              f'графика нагрузок, в котором вечерний максимум будет ниже утреннего максимума ' \
                              f'активной мощности.'
            else:
                temp_cons_2 = ''

            temp_cons_3 = f'3. {temp_cons_2} При выполнении условия оплаты за электроэнергию по ДД-тарифу переход на ' \
                          f'двухставочную форму оплаты приведет ' \
                        f'к увеличению стоимости электроэнергии за рассмотренный период на {temp_pay} руб. \n\t' \
                        f'4. Для повышения эффективности использования дифференцированной оплаты необходимо реализовать ' \
                        f'ряд условий: контролировать не превышение вечернего пика над утренним; ' \
                        f'обеспечить снижения активной мощности в пиковой тарифной зоне, где действует ' \
                        f'максимальная оплата электроэнергии; загрузить производственные мощности в ночной зоне, ' \
                        f'где действует минимальная оплата за электроэнергию.'
            conclusion3 = conclusion3 + temp_cons_3
        if base_tariff == 0:
            temp_pay = float(df_pay[df_pay.iloc[:, 5] == 0]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - pay_opt
            temp_cons = f'3. Проведенный анализ отражает не эффективность оплаты электроэнергии по двухставочному тарифу ' \
                        f'с заявленной мощностью в общем случае годовая разница стоимости электроэнергии между ' \
                        f'существующим тарифом и оптимальным достигает {temp_pay} руб. \n\t'
            conclusion3 = conclusion3 + temp_cons
    elif base_tariff == ind_opt:
        conclusion3 = f'1. Существующий тариф оплаты за электрическую энергию ({tar_exist}) является оптимальным. \n\t' \
                      f'2. При существующем тарифе достигается минимальная стоимость 1 кВтч равная ' \
                      f'{energy_unit_cost_opt} руб/кВтч. \n\t'
        if base_tariff == 1:
            temp_pay = float(df_pay[df_pay.iloc[:, 5] == 3]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - pay_opt
            temp_cons = f'3. Переход на дифференцированную форму оплаты за электроэнергию даже в случае полного ' \
                        f'контроля не превышения вечернего минимума над утренним приведет к увеличению ' \
                        f'стоимости электроэнергии на {temp_pay} руб.\n\t' \
                        f'4. Переход к дифференцированной форме оплаты будет эффективен только в случае значительного ' \
                        f'изменения графика работы предприятия.'
            conclusion3 = conclusion3 + temp_cons
        if base_tariff == 2:
            temp_pay = float(df_pay[df_pay.iloc[:, 5] == 1]
                             ['Суммарная оплата за электроэнергию и мощность, руб.']) - pay_opt
            temp_cons = f'3. Переход на двухставочную форму оплаты экономически не эффективен поскольку приведен ' \
                        f'к увеличению стоимости электроэнергии на {temp_pay} руб.\n\t' \
                        f'4. Для повышения эффективности использования дифференцированной оплаты необходимо реализовать ' \
                        f'ряд условий: контролировать не превышение вечернего пика над утренним; ' \
                        f'обеспечить снижения активной мощности в пиковой тарифной зоне, где действует ' \
                        f'максимальная оплата электроэнергии; загрузить производственные мощности в ночной зоне, ' \
                        f'где действует минимальная оплата за электроэнергию.'
            conclusion3 = conclusion3 + temp_cons
    else:
        conclusion3 = ''
    context['conclusion3'] = conclusion3
    print(conclusion3)
    # Рисунок 6.1 - Диаграмма сравнения тарифов оплаты за электроэнергию и мощность
    df_pay_for_plot = pd.DataFrame({'Тариф оплаты за электроэнергию':
                                        ['Д-тариф \n (заявленная мощность)',
                                         'Д-тариф \n (фактическая мощность)',
                                         'ДД-тариф \n (без регулирования)',
                                         'ДД-тариф \n (с регулированием)'],
                                    'Оплата за электрическую мощность, руб': [d_tariff_declared.sum_pay_power,
                                                                              d_tariff.sum_pay_power,
                                                                              dif_tariff.sum_pay_power,
                                                                              dif_tariff_reg.sum_pay_power],
                                    'Оплата за электрическую энергию, руб': [d_tariff_declared.sum_pay_energy,
                                                                             d_tariff.sum_pay_energy,
                                                                             dif_tariff.sum_pay_energy,
                                                                             dif_tariff_reg.sum_pay_energy],
                                    'Суммарная оплата за электроэнергию и мощность, руб.':
                                        [d_tariff_declared.sum_pay_power_and_energy, d_tariff.sum_pay_power_and_energy,
                                         dif_tariff.sum_pay_power_and_energy, dif_tariff_reg.sum_pay_power_and_energy],
                                    })

    df_pay_for_plot = df_pay_for_plot.set_index('Тариф оплаты за электроэнергию')
    image61= fgr.my_bar(template=template, data=df_pay_for_plot, name_fig='Сравнение тарифов')
    context['СравнениеДиаграмма'] = image61

    # Таблица 6.3 – Результаты сравнения тарифов оплаты за электрическую мощность и энергию
    df_tariff_declared = d_tariff_declared.df_pay_energy_month.iloc[:, [4, 7, 8]]
    df_d_tariff = d_tariff.df_pay_energy_month.iloc[:, [4, 7, 8]]
    df_dif_tariff = dif_tariff.df_pay_energy_month.iloc[:, [13, 14, 15, 16, 17, 18]]
    df_d_tariff_reg = dif_tariff_reg.df_pay_energy_month.iloc[:, [13, 14, 15, 16, 17, 18]]
    df_total = pd.concat([df_tariff_declared, df_d_tariff, df_dif_tariff, df_d_tariff_reg], axis=1)
    table_contents_6_3 = tbl.table_dynamic(data=df_total.reset_index())
    context['table_contents_6_3'] = table_contents_6_3

    # Таблица П3 – Результаты сравнения эффективности тарифов по суткам
    df_tariff_declared = d_tariff_declared.df_pay_energy_day.iloc[:, [4, 7, 8]].reset_index().set_index(
        'Период наблюдений')
    df_d_tariff = d_tariff.df_pay_energy_day.iloc[:, [4, 7, 8]].reset_index().set_index('Период наблюдений').drop(
        ['День'], axis=1)
    df_dif_tariff = dif_tariff.df_pay_energy_day.iloc[:, [17, 18, 19, 20, 21, 22]]
    df_d_tariff_reg = dif_tariff_reg.df_pay_energy_day.iloc[:, [17, 18, 19, 20, 21, 22]]
    df_total_day = pd.concat([df_tariff_declared, df_d_tariff, df_dif_tariff, df_d_tariff_reg], axis=1)
    table_contents_P_3 = tbl.table_dynamic(data=df_total_day.reset_index())
    context['table_contents_P3'] = table_contents_P_3


    # #_____________ГЛАВА 7 ОЦЕНКА ЭФФЕКТИВНОСТИ СМЕЩЕНИЯ ПРОИЗВОДСТВЕННОГО ЦИКЛА _______________№
    result_analytic_step = analytic.step_thread(df_input_statistics=df1, df_declared_power=df3,
                                                  ab=ab, bb=bb, kt=kt, kb=kb, shift_left = -6, shift_right = 6)
    # Таблица 7.1 – Результаты сравнения годовой
    table_contents_7_1 = tbl.table_dynamic(data=result_analytic_step.reset_index())
    context['table_contents_7_1'] = table_contents_7_1
    print(result_analytic_step)

    # РИСУНОК 7.1 ШАГ СМЕЩЕНИЯ ЦИКЛА

    # ВЫБИРАЕМ ДАННЫЕ РЕЗУЛЬТАРУЩЕЙ СУММЫ ОПЛАТЫ ОТ ШАГА СМЕЩЕНИЯ

    result_analytic_step_total = result_analytic_step.iloc[:, [2, 5, 11, 17]]

    ylim_min = result_analytic_step_total.iloc[:,3].min() * 0.8
    image71 = fgr.my_bar(template=template, data=result_analytic_step_total, name_fig='Смещение цикла',
                         ylim_min = ylim_min, loc='upper right', bbox_to_anchor=(1, 0.5), fontsize=6)

    context['Смещение_Цикла'] = image71
    pay_tariff_declared_step = result_analytic_step_total.iloc[:, [0]].min().values[0]
    pay_d_tariff_step = result_analytic_step_total.iloc[:, [1]].min().values[0]
    pay_dif_d_tariff_step = result_analytic_step_total.iloc[:, [2]].min().values[0]
    pay_dif_d_tariff_reg_step = result_analytic_step_total.iloc[:, [3]].min().values[0]

    context['плата_заявл'] = pay_tariff_declared_step
    context['плата_факт'] = pay_d_tariff_step
    context['плата_диф'] = pay_dif_d_tariff_step
    context['плата_диф_рег'] = pay_dif_d_tariff_reg_step


    step_min_declared = result_analytic_step_total[result_analytic_step_total.iloc[:, 0] ==
                                                   pay_tariff_declared_step].index.values[0]
    step_min_d_tariff = result_analytic_step_total[result_analytic_step_total.iloc[:, 1] ==
                                                   pay_d_tariff_step].index.values[0]
    step_min_dif_tariff = result_analytic_step_total[result_analytic_step_total.iloc[:, 2] ==
                                                   pay_dif_d_tariff_step].index.values[0]
    step_min_dif_tariff_reg = result_analytic_step_total[result_analytic_step_total.iloc[:, 3] ==
                                                   pay_dif_d_tariff_reg_step].index.values[0]
    context['цикл_заявл'] = step_min_declared
    context['цикл_факт'] = step_min_d_tariff
    context['цикл_диф'] = step_min_dif_tariff
    context['цикл_диф_рег'] = step_min_dif_tariff_reg

    df_new = pd.DataFrame({'Оптимальный шаг смещения, ч': [step_min_declared, step_min_d_tariff,
                                                        step_min_dif_tariff, step_min_dif_tariff_reg],
                           'Оплата за электроэнергию при оптимальном шаге, руб.': [pay_tariff_declared_step,
                                                                                   pay_d_tariff_step,
                                                                                   pay_dif_d_tariff_step,
                                                                                   pay_dif_d_tariff_reg_step
                                                                                   ],
                           'Вид тарифа': ['двухставочный тариф с оплатой за заявленную мощность',
                                          'двухставочный тариф с оплатой за фактическую мощность',
                                          'двухставочно-дифференцированный тариф без изменения графика нагрузок',
                                          'двухставочно-дифференцированный тариф при регулировании не превышения '
                                          'вечернего максимума над утренним',
                                          ]
                           })
    pay_min_step = df_new.iloc[:, [1]].min().values[0]
    context['оптим_оплата'] = pay_min_step

    step_opt = df_new[df_new.iloc[:, 1] == pay_min_step].iloc[:, 0].values[0]

    context['оптим_смещ'] = step_opt

    opt_type = df_new[df_new.iloc[:, 1] == pay_min_step].iloc[:, 2].values[0]
    context['опт_тар_смещ'] = opt_type

    context['сущ_оплата'] = pay_exist
    context['разница_платы'] = round((pay_exist - pay_min_step), 1)
    context['оптим_процент'] = round((pay_exist / pay_min_step - 1) * 100, 1)


    # ____________ВИЗУАЛИАЦИЯ ОТЧЕТА_____________________#
    template.render(context)
    name_template = f'Отчет {str(company_name)}.docx'
    template.save(name_template)

    return conclusion3




