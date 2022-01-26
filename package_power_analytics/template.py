import random
import pandas as pd
import datetime
from io import BytesIO
import base64
import traceback


def edit_style(writer):
    workbook = writer.book
    worksheet1 = writer.sheets['Исх данные']
    fmt = workbook.add_format({'font_name': 'Times New Roman', 'font_size': '12', 'text_wrap': 'True'})
    worksheet1.set_column('A:A', 70, fmt)
    worksheet1.set_column('B:B', 30, fmt)

    # Здесь можно посмотреть форматирование эксель https://russianblogs.com/article/9047967407/
    format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
    worksheet1.conditional_format('A1:A10', {'type': 'no_blanks', 'format': format})
    worksheet1.conditional_format('B1:B10', {'type': 'no_blanks', 'format': format})

    worksheet2 = writer.sheets['Заявл мощность']
    worksheet2.set_column('A:C', 25, fmt)
    worksheet2.conditional_format('A1:C13', {'type': 'no_blanks', 'format': format})

    worksheet3 = writer.sheets['Получасовая статистика']
    worksheet3.set_column('A:C', 25, fmt)
    worksheet3.conditional_format('A1:C500', {'type': 'no_blanks', 'format': format})


#Создание шаблона Эксель файла для ввода исходных данных
def create_template_input_data(start_time, finish_time, df1=None, df2=None, df3=None):

    if df1 is None:

        date = pd.date_range(start=f'{start_time.year}-{start_time.month}-{start_time.day}',
                             end=f'{finish_time.year}-{finish_time.month}-{finish_time.day}', freq='30min')[:-1]
        active_power = [random.randint(0, 200) for _ in range(len(date))]
        reactive_power = [random.randint(0, 200) for _ in range(len(date))]
        df1 = pd.DataFrame(
            {'Дата': date, 'Активная мощность, кВт': active_power, 'Реактивная мощность, кВАр': reactive_power})
        file_name_df1 = 'Анализируемая статистика.xlsx'
        df1.to_excel(file_name_df1, index=False, sheet_name='Исходная статистика')

        number_of_month = [i for i in range(1, 13, 1)]
        name_month = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь',
                      'Ноябрь', 'Декабрь']
        declared_power = ['' for _ in range(len(name_month))]
        df2 = pd.DataFrame(
            {'Номер месяца': number_of_month, 'Месяц': name_month, 'Заявленная мощность, кВт': declared_power})

        name_date = ['Название предприятия',
                     'Место сбора групповых графиков нагрузок',
                     'Базовый курс доллара США в соответствии с декларацией о тарифах, Кб, руб/долл. США',
                     'Текущий курс доллара, принимаемый в анализе, Кт, руб/долл. США',
                     'Основная плата - за мощность (на 1 месяц), а, руб/кВт',
                     'Дополнительная плата - за энергию , b, руб/кВтч',
                     'Дата ввода декларации тарифов',
                     'Дата курса доллара по отношению к белорускому рубля по Нацбанку РБ, руб/долл. США',
                     'Индикатор существующего тирифа оплаты за ЭЭ:\n'
                     '0 - двухставочный с заявленной мощностью\n'
                     '1 - двухставочный с фактической мощностью\n'
                     '2 - двухставочно-дифференцированный тариф'
                     ]

        res_date = ['Предприятие "А"',
                    'Сумма нагрузок',
                    float('2.5789'),
                    float('2.5118'),
                    float('26.71339'),
                    float('0.22591'),
                    datetime.date(year=datetime.datetime.now().year, month=1, day=1),
                    datetime.date(year=datetime.datetime.now().year, month=8, day=13),
                    int('2')
                    ]
        df3 = pd.DataFrame(
            {'Наименование показателя': name_date, 'Значение': res_date})

        buffer = BytesIO()
        writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
        df3.to_excel(writer, sheet_name='Исх данные', index=False)
        df2.to_excel(writer, sheet_name='Заявл мощность', index=False)
        df1.to_excel(writer, sheet_name='Получасовая статистика', index=False)
        edit_style(writer)
        writer.save()
        processed_data = buffer.getvalue()
    else:

        buffer = BytesIO()
        writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
        df3.to_excel(writer, sheet_name='Исх данные', index=False)
        df2.to_excel(writer, sheet_name='Заявл мощность', index=False)
        df1.to_excel(writer, sheet_name='Получасовая статистика', index=False)
        edit_style(writer)
        writer.save()
        processed_data = buffer.getvalue()

    return processed_data


#функция создает данные в буфере обмена
def write_to_excel(*args):
    """Исходные данные датафреймы (через запятую)"""
    buffer = BytesIO()
    writer = pd.ExcelWriter(buffer, engine='xlsxwriter')
    for count, df in enumerate(args):
        df.to_excel(writer, sheet_name='Лист '+str(count), index=False)
    writer.save()
    processed_data = buffer.getvalue()
    return processed_data

