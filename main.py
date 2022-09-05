import streamlit as st
import pandas as pd
from package_power_analytics.template import create_template_input_data, write_to_excel
from package_power_analytics.template import create_template_input_data, write_to_excel
import package_power_analytics.myplotly as mp
import package_power_analytics.analytic as an
import os, sys
import datetime
import time
import traceback
from io import BytesIO
import base64
import numpy as np
import sqlite3
# import sweetviz as sv
# from pandas_profiling import ProfileReport


# Как развернуть приложение можно почитать здесь
# https://www.analyticsvidhya.com/blog/2021/06/deploy-your-ml-dl-streamlit-application-on-heroku/


# Функция позволяющая сделать download на выгрузку данных расчета
# list_data_frame
@st.cache
def get_table_download_link(*args):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  *args - через запятую поданные количество датафреймов для записи в файл
    out: href string
    """
    val = write_to_excel(args)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Result_cluster.xlsx">' \
           f'Скачать xlsx файл результата</a>'  # decode b'abc' => abc

def list_times(hour_start=0, hour_finish=24, minute_delta=30):
    count = (hour_finish - hour_start) * 60
    res = [(datetime.datetime.combine(datetime.date.today(), datetime.time(hour=hour_start,minute=0)) +
            datetime.timedelta(minutes=i)).time().strftime("%H:%M")
           for i in range(0, count , minute_delta)]
    return res


def app(mode='not_demo'):
    st.sidebar.markdown('## Генерация шаблона данных')
    agree = st.sidebar.checkbox('Включить редактор шаблона')
    if agree:
        year_now = datetime.datetime.now().year
        year_last = year_now - 1
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_time = st.sidebar.date_input('Начальная дата наблюдений', value=datetime.date(year_last, 1, 1))
        with col2:
            finish_time = st.sidebar.date_input('Конечная дата наблюдений', value=datetime.date(year_last + 1, 1, 1))

        form_data = st.sidebar.radio(
            "Вид исходной статистики",
            ('Получасовая энергия', 'Получасовая мощность'))
        if form_data == 'Получасовая энергия':
            type_stat = 0
        else:
            type_stat = 1


        file_xlsx = create_template_input_data(start_time, finish_time, type_stat=type_stat)
        st.sidebar.download_button(label='Сгенерировать шаблон для ввода данных Excel-файла',
                                   data=file_xlsx,
                                   file_name='Шаблон ввода исходных данных.xlsx')

    st.sidebar.write('________')

    try:
        with st.expander('ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ ПРОЕКТА'):
            if mode == 'demo':
                uploaded_file = ('Тестовые данные.xlsx')
            else:
                uploaded_file = st.file_uploader(label='Загрузите или перетащите файл Excel ', type=['xlsx', 'xls'])

            if uploaded_file is None:
                st.markdown('Необходимо загрузить шаблон исходных данных.')
                raise Exception
            else:
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                base_sheets = ['Получасовая статистика', 'Заявл мощность', 'Исх данные']
                for sheet in base_sheets:
                    if sheet not in sheet_names:
                        st.subheader('Выбранный шаблон не соответствует базовому. '
                                     'Скачайте и заполните шаблон для ввода данных.')
                        raise Exception

                # Заполнение информационного поля ввода исходных данных
                df_initial_data = xls.parse('Исх данные')
                agree = st.checkbox('Показать вспомогательные данные')
                if agree:
                    st.markdown('##### Исходные данные')

                    for row_index, row in df_initial_data.iterrows():
                        st.markdown(f'###### {row[0]}')
                        if row[1] is not np.nan:
                            st.text(row[1])
                        else:
                            st.markdown('Отсутствуют данные, которые должны быть введены в загруженном шаблоне.')
                            raise Exception

                # Проверка на наличие данных заявленной мощности
                df_declared = xls.parse('Заявл мощность')

                for row_index, row in df_declared.iterrows():
                    if row[2] is not np.nan:
                        pass
                    else:
                        st.markdown('Отсутсвуют данные заявленной мощности')
                        raise Exception

                st.write('Исходная статистика')
                df_power_statistics = xls.parse('Получасовая статистика')
                st.write(df_power_statistics)

                if df_power_statistics.isnull().values.any() == True:
                    st.markdown('**Получасовая статистика активной и реактивной мощности не должна'
                                'содержать пропусков!**')
                    raise Exception('Ошибка. Не заполнены данные в исходной статистике.')

                st.markdown('###### Входная статистика:')
                st.write(f'1. {df_power_statistics.columns[1]}')
                st.write(f'2. {df_power_statistics.columns[2]}')
                if df_power_statistics.columns[1] == "Активная энергия (А+), кВтч":
                    df_power_statistics[df_power_statistics.columns[1]] = \
                        df_power_statistics[df_power_statistics.columns[1]].apply(lambda x: x * 2)
                    df_power_statistics[df_power_statistics.columns[2]] = \
                        df_power_statistics[df_power_statistics.columns[2]].apply(lambda x: x * 2)
                    df_power_statistics = df_power_statistics. \
                        rename(columns={df_power_statistics.columns[1]: 'Активная мощность, кВт',
                                        df_power_statistics.columns[2]: 'Реактивная мощность, кВар'})

        # Ручное редактирование
        st.sidebar.header('Настройки расчета')
        st.sidebar.markdown('#### Редактирование исходных данных')
        agree = st.sidebar.checkbox('Включить ручное редактирование исходных данных')
        if agree:

            df_initial_data.iloc[0, 1] = st.sidebar.text_input(label='Название предприятия',
                                                               value=df_initial_data.iloc[0, 1])
            df_initial_data.iloc[2, 1] = st.sidebar.text_input(label='Базовый курс доллара США, Кб, руб/долл. США',
                                                               value=float(df_initial_data.iloc[2, 1]))
            df_initial_data.iloc[3, 1] = st.sidebar.text_input(label='Текущий курс доллара, Кт, руб/долл. США',
                                                               value=float(df_initial_data.iloc[3, 1]))
            df_initial_data.iloc[4, 1] = st.sidebar.text_input(label='Основная плата - за мощность, а, руб/(кВт*мес)',
                                                               value=float(df_initial_data.iloc[4, 1]))
            df_initial_data.iloc[5, 1] = st.sidebar.text_input(label='Дополнительная плата - за энергию , b, руб/кВтч',
                                                               value=float(df_initial_data.iloc[5, 1]))
            option = st.sidebar.selectbox(label='Существующий тариф оплаты',
                                          options=('Д-тариф с заявленной мощностью',
                                                   'Д-тариф с фактической мощностью',
                                                   'ДД-тариф'), index=df_initial_data.iloc[8, 1])

            if option == 'Д-тариф с заявленной мощностью':
                df_initial_data.iloc[8, 1] = int(0)
            elif option == 'Д-тариф с фактической мощностью':
                df_initial_data.iloc[8, 1] = int(1)
            elif option == 'ДД-тариф':
                df_initial_data.iloc[8, 1] = int(2)

            declared_power = st.sidebar.text_input(label='Заявленная мощность, кВт/мес',
                                                   value=float(df_declared.iloc[1, 2]))
            df_declared.iloc[:, 2] = float(declared_power)

            file_xlsx_new = create_template_input_data(start_time=None, finish_time=None, df1=df_power_statistics,
                                                       df2=df_declared, df3=df_initial_data)

            st.sidebar.download_button(label='Сохранить изменения в Excel-файл',
                                       data=file_xlsx_new,
                                       file_name=f'Исходные данные {df_initial_data.iloc[0, 1]}.xlsx')

        st.sidebar.write('________')

        st.sidebar.markdown('#### Первичный анализ статистики')

        options_multiselect = st.sidebar.multiselect(
            'РАСЧЕТНЫЕ МОДУЛИ',
            ['Исходные данные', 'Описательная статистика', 'Часовые графики нагрузок'],
            [])


        if len(options_multiselect) !=0:
            st.subheader('Первичный анализ исходных статистических данных')

        if 'Исходные данные' in options_multiselect:
            st.markdown('#### Статистические данные получасовой мощности')
            st.write('Приведенные ниже статистические данные представлены в форме получасовой мощности '
                     'над которыми в дальнейшем проводятся расчеты.')
            st.write(df_power_statistics)
        if 'Описательная статистика' in options_multiselect:
            st.markdown('#### Описательная статистика')
            st.write('Описательная статистика позволяет обобщать первичные результаты, '
                     'полученные при наблюдении или в эксперименте. В качестве статистических показателей '
                     'используются: среднее, медиана, мода, дисперсия, стандартное отклонение и др.')
            res1 = an.describe_statistics(one_d_df=df_power_statistics.iloc[:, [1]])
            st.markdown(f'###### {df_power_statistics.iloc[:, [1]].columns[0]}')
            st.write(res1)

            res2 = an.describe_statistics(one_d_df=df_power_statistics.iloc[:, [2]])
            st.markdown(f'###### {df_power_statistics.iloc[:, [2]].columns[0]}')
            st.write(res2)
            # Запись результатов в файл
            file_xlsx = write_to_excel(res1, res2)
            st.download_button(label='Сохранить результаты в xlsx файл',
                               data=file_xlsx,
                               file_name='Результаты описательной статистики.xlsx')
        if 'Часовые графики нагрузок' in options_multiselect:
            st.markdown('#### Получасовые и сглаженные графики изменения мощности')
            # График изменения активной мощности
            dfx = df_power_statistics.iloc[:, [0]]
            dfy_act = df_power_statistics.iloc[:, [1]]
            fig = mp.myplotly(dfx, dfy_act)
            st.markdown('##### Активная мощность')
            st.write(fig)

            # График изменения реактивной мощности
            dfx = df_power_statistics.iloc[:, [0]]
            dfy_react = df_power_statistics.iloc[:, [2]]
            fig = mp.myplotly(dfx, dfy_react)
            st.write('_____')
            st.markdown('##### Реактивная мощность')
            st.write(fig)

            # График изменения коэффициента мощности
            power_coefficients_fi = df_power_statistics
            power_coefficients_fi['Полная мощность, кВА'] = \
                round(np.sqrt(df_power_statistics.iloc[:, 1] ** 2 + df_power_statistics.iloc[:, 2] ** 2))
            power_coefficients_fi['Коэффициент активной мощности (cosф)'] = df_power_statistics.iloc[:, 1] / \
                                                                            df_power_statistics.iloc[:, 3]

            dfx = power_coefficients_fi.iloc[:, [0]]
            dfy_react = power_coefficients_fi.iloc[:, [4]]
            fig = mp.myplotly(dfx, dfy_react)
            st.write('_____')
            st.markdown('##### Коэффициент мощности')
            st.write(fig)
            st.write('_____')

        # ______________________#АНАЛИЗ ГРАФИКОВ НАГРУЗОК#______________________#______________________#______________________#

        st.sidebar.markdown('#### Анализ графика нагрузок')
        check_coefficients = st.sidebar.checkbox('Выполнить анализ ГЭН')
        st.sidebar.write('____')
        # Делаем предварительные расчеты
        power_coefficients = an.PowerGraphCoefficients(df=df_power_statistics)
        df_mean = power_coefficients.calculation_mean_power_of_month()
        df_square = power_coefficients.calculation_square_power_of_month()
        df_max = power_coefficients.calculation_max_power_of_month()
        df_coef_fill = power_coefficients.coefficient_fill()
        df_coef_shape = power_coefficients.coefficient_shape()
        df_coef_fi = power_coefficients.coefficient_fi()
        df_coef_max = power_coefficients.coefficient_max()

        if check_coefficients:
            st.subheader('Определение основных физических величин')
            st.write(
                'К физическим величинам, характеризующим графики электрических нагрузок, относится средняя, '
                'среднеквадратичная и максимальная нагрузка.')
            check_mean_power = st.checkbox('Расчет средней месячной нагрузки')
            if check_mean_power:
                st.markdown('#### Средняя нагрузка')
                st.write(
                    'Средняя нагрузка – это постоянная, неизменная величина за любой рассматриваемый промежуток '
                    'времени, которая вызывает такой же расход электроэнергии, как и изменяющаяся '
                    'за это время нагрузка:')

                st.latex(r'''P_с =  \frac {\sum_{i=1}^{n} P_{сi} }  {N}''')

                df_mean_str = df_mean.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_mean_str)

                dfx = df_mean_str.iloc[:, [0]]
                dfy_mean_act = df_mean.iloc[:, [1]]
                dfy_mean_react = df_mean.iloc[:, [2]]
                dfy_mean_full = df_mean.iloc[:, [3]]
                fig = mp.my_histogram(dfx, dfy_mean_act, dfy_mean_react, dfy_mean_full, )
                st.markdown('##### Изменение средней нагрузки')
                st.write(fig)

                act_mean = round(df_mean.iloc[:, 1].mean(), 2)
                react_mean = round(df_mean.iloc[:, 2].mean(), 2)
                full_mean = round(df_mean.iloc[:, 3].mean(), 2)

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Cредняя активная нагрузка', value=f'{act_mean} кВт')
                st.metric(label='Cредняя реактивная нагрузка', value=f'{react_mean} кВар')
                st.metric(label='Cредняя полная нагрузка', value=f'{full_mean} кВА')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_mean)
                st.download_button(label='Сохранить результаты средней нагрузки в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета средней нагрузки.xlsx')
                st.write('__________')

            # Расчет среднеквадратичной нагрузки
            check_square_power = st.checkbox('Расчет среднеквадратичной месячной нагрузки')
            if check_square_power:
                st.subheader('Среднеквадратичная месячная нагрузка')
                st.write(
                    'Среднеквадратичная нагрузка – это постоянная, неизменная нагрузка за любой рассматриваемый'
                    ' промежуток времени, которая обуславливает такие же потери мощности в проводниках, '
                    'как и изменяющаяся за это время нагрузка.')
                st.latex(r'''P_{ск} = \sqrt { \frac  {\sum_{i=1}^{n} P^{2}_{сi} }  {N} }''')

                df_square_str = df_square.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_square_str)

                dfx = df_square_str.iloc[:, [0]]
                dfy_square_act = df_square.iloc[:, [1]]
                dfy_square_react = df_square.iloc[:, [2]]
                dfy_square_full = df_square.iloc[:, [3]]
                fig = mp.my_histogram(dfx, dfy_square_act, dfy_square_react, dfy_square_full)
                st.markdown('##### Изменение среднеквадратической нагрузки')
                st.write(fig)

                act_square = round(df_square.iloc[:, 1].mean(), 2)
                react_square = round(df_square.iloc[:, 2].mean(), 2)
                full_square = round(df_square.iloc[:, 3].mean(), 2)

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Cреднеквадратическая активная нагрузка', value=f'{act_square} кВт')
                st.metric(label='Cреднеквадратическая реактивная нагрузка', value=f'{react_square} кВар')
                st.metric(label='Cреднеквадратическая полная нагрузка', value=f'{full_square} кВА')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_square)
                st.download_button(label='Сохранить результаты среднеквадратической нагрузки в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета среднеквадратичной нагрузки.xlsx')
                st.write('____')

            # Расчет максимальной месячной нагрузки
            check_max_power = st.checkbox('Расчет максимальной месячной нагрузки')
            if check_max_power:
                st.subheader('Максимальная месячная нагрузка')
                st.write(
                    'Максимальная нагрузка – это наибольшая из средних нагрузок за '
                    'рассматриваемый промежуток времени.')
                st.latex(r'''P_{max} = max(P_i)''')

                df_max_str = df_max.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_max_str)

                dfx = df_max_str.iloc[:, [0]]
                dfy_max_act = df_max.iloc[:, [1]]
                dfy_max_react = df_max.iloc[:, [2]]
                dfy_max_full = df_max.iloc[:, [3]]
                fig = mp.my_histogram(dfx, dfy_max_act, dfy_max_react, dfy_max_full)

                st.markdown('##### Изменение максимальной нагрузки')
                st.write(fig)

                act_max = round(df_max.iloc[:, 1].max(), 2)
                react_max = round(df_max.iloc[:, 2].max(), 2)
                full_max = round(df_max.iloc[:, 3].max(), 2)

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Максимальная активная нагрузка', value=f'{act_max} кВт')
                st.metric(label='Максимальная реактивная нагрузка', value=f'{react_max} кВар')
                st.metric(label='Максимальная полная нагрузка', value=f'{full_max} кВА')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_max)
                st.download_button(label='Сохранить результаты максимальной нагрузки в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета максимальной нагрузки.xlsx')
                st.write('___')

            st.subheader('Определение безразмерных физических величин графиков нагрузки')
            st.write(
                'Наряду с физическими величинами графики нагрузки описываются безразмерными коэффициентами. '
                'Эти коэффициенты устанавливают связь между основными физическими величинами, характеризуют '
                'неравномерность графиков нагрузки, а также использование электроприёмников и потребителей '
                'электроэнергии по мощности и времени.')

            # Коэффициент максимума график
            check_cof_max_power = st.checkbox('Расчет коэффициента максимума графика')

            if check_cof_max_power:
                st.subheader('Коэффициент максимума графика нагрузок')


                st.latex(r'''K_{м.г} = \frac  {P_м}  {P_{ср}} ''')

                df_coef_max_str = df_coef_max.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_coef_max_str)

                dfx = df_coef_max_str.iloc[:, [0]]
                dfy_coef_max = df_coef_max.iloc[:, [1]]
                fig = mp.my_histogram(dfx, dfy_coef_max)
                st.markdown('##### Изменение коэффициента максимума')
                st.write(fig)

                max_coef = round(df_coef_max.iloc[:, 1].max(), 3)
                df_temp = df_coef_max.loc[df_coef_max.iloc[:, 1] == df_coef_max.iloc[:, 1].max()]
                max_mohtn_coef = df_temp.iloc[:, 0].values[0]

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Наибольшее значение коэффициента максимума ', value=f'{max_coef}')
                st.metric(label='Месяц с наибольшим значением коэффициента максимума', value=f'{max_mohtn_coef}')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_coef_max)
                st.download_button(label='Сохранить результаты расчета коэффициента максимума в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета коэффициента максимума.xlsx')

                st.write('____')

            # Коэффициент заполенния графика
            check_coef_fill_power = st.checkbox('Расчет коэффициента заполнения графика')
            if check_coef_fill_power:
                st.subheader('Коэффициент заполнения графика нагрузок')
                st.write(
                    'Характеризует неравномерность графика нагрузок. Для рациональной передачи электроэнергии, '
                    'т.е. снижения к минимуму потерь в питающих и распределительных сетях предприятия, '
                    'коэффициент заполнения графика должен быть приближен к единице, '
                    'что определяется равномерностью включения технологических потребителей электроэнергии')

                st.latex(r'''K_{з.г} = \frac  {P_{ср}}  {P_{м}} ''')

                df_coef_fill_str = df_coef_fill.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_coef_fill_str)

                dfx = df_coef_fill_str.iloc[:, [0]]
                dfy_coef_fill = df_coef_fill.iloc[:, [1]]
                fig = mp.my_histogram(dfx, dfy_coef_fill)
                st.markdown('##### Изменение коэффициента заполнения')
                st.write(fig)

                max_coef_fill = round(df_coef_fill.iloc[:, 1].max(), 3)
                df_temp = df_coef_fill.loc[df_coef_fill.iloc[:, 1] == df_coef_fill.iloc[:, 1].max()]
                max_mohtn_coef = df_temp.iloc[:, 0].values[0]

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Наибольшее значение коэффициента заполнения ', value=f'{max_coef_fill}')
                st.metric(label='Месяц с наибольшим значением коэффициента заполнения', value=f'{max_mohtn_coef}')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_coef_fill)
                st.download_button(label='Сохранить результаты расчета коэффициента заполнения в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета расчета коэффициента заполнения.xlsx')

                st.write('____')

            # Коэффициент формы графика
            check_shape_power = st.checkbox('Расчет коэффициента формы графика')
            if check_shape_power:
                st.subheader('Коэффициент формы графика нагрузок')
                st.write('Характеризует неравномерность график нагрузок. Для минимизации потерь электроэнергии '
                         'в питающих, распределительных сетях и трансформаторах системы существующей системы '
                         'электроснабжения коэффициент формы графика должен быть приближен к единице.')

                st.latex(r'''K_{ф.г} = \frac  {P_{ск}}  {P_{с}} ''')

                df_coef_shape_str = df_coef_shape.astype(str)
                st.markdown('##### Результаты расчета')
                st.write(df_coef_shape_str)

                dfx = df_coef_shape_str.iloc[:, [0]]
                dfy_coef_fill = df_coef_shape.iloc[:, [1]]
                fig = mp.my_histogram(dfx, dfy_coef_fill)
                st.markdown('##### Изменение коэффициента формы графика нагрузок')
                st.write(fig)

                max_coef_shape = round(df_coef_shape.iloc[:, 1].max(), 3)
                df_temp = df_coef_shape.loc[df_coef_shape.iloc[:, 1] == df_coef_shape.iloc[:, 1].max()]
                max_mohtn_shape = df_temp.iloc[:, 0].values[0]

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Наибольшее значение коэффициента формы ', value=f'{max_coef_shape}')
                st.metric(label='Месяц с наибольшим значением коэффициента формы', value=f'{max_mohtn_shape}')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_coef_shape)
                st.download_button(label='Сохранить результаты расчета коэффициента формы в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета коэффициента формы.xlsx')
                st.write('____')

            # Коэффициент активной мощности
            check_coefficient_fi = st.checkbox('Расчет коэффициента активной мощности')
            if check_coefficient_fi:
                st.subheader('Коэффициент активной мощности')
                st.write('Коэффициент активной мощности – безразмерная физическая величина, '
                         'характеризующая потребителя переменного электрического тока с точки зрения '
                         'наличия в нагрузке реактивной составляющей.')

                st.latex(r'''\cos {\phi} = \frac  {P_{ср}}  {S_{ср}} ''')

                df_coef_fi_str = df_coef_fi.astype(str)

                st.markdown('##### Результаты расчета')

                dfx = df_coef_fi.iloc[:, [0]].astype(str)
                dfy_coef_fi = df_coef_fi.iloc[:, [1]].astype(float)

                fig = mp.my_histogram(dfx, dfy_coef_fi)
                st.write(fig)

                max_coef_fi = round(df_coef_fi.iloc[:, 1].max(), 3)
                min_coef_fi = round(df_coef_fi.iloc[:, 1].min(), 3)

                st.subheader('Наиболее значимые результаты')
                st.metric(label='Наибольшее значение коэффициента мощности ', value=f'{max_coef_fi}')
                st.metric(label='Наименьшее значение коэффициента мощности', value=f'{min_coef_fi}')

                # Запись результатов в файл
                file_xlsx = write_to_excel(df_coef_fi)
                st.download_button(label='Сохранить результаты коэффициента мощности в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета коэффициента мощности.xlsx')
                st.write('____')

            # Запись всех результатов анализа графика в файл
            write_to_exel_buffer = power_coefficients.write_to_exel_buffer()
            st.download_button(label='Сохранить групповые результаты в xlsx файл',
                               data=write_to_exel_buffer,
                               file_name='Результаты расчета показателей ГЭН.xlsx')
            st.write('____')

        # ______________________#АНАЛИЗ ДВУХСТАВОЧНОГО ТАРИФА С ЗАЯВЛЕННОЙ МОЩНОСТЬЮ#______________________#___________________#

        # УПРАВЛЕНИЕ ИЗМЕНЕНИЕМ ГРАНИЦЫ МАКСИМУМОВ

        st.sidebar.markdown('#### Границы максимумов энергосистемы и лимиты мощности')
        agree = st.sidebar.checkbox('Изменить границы максимумов')
        time_start_morning, time_finish_morning, time_start_evening, time_finish_evening = None, None, None, None
        if agree:
            # Список времени в в строках 00:00, 00:30... str
            list_t = list_times(hour_start=0, hour_finish=15, minute_delta=30)
            st.sidebar.markdown('#### Утренний максимум')
            ######  время в часах начала утреннего максимума
            select_dtm_start = datetime.datetime. \
                strptime(st.sidebar.selectbox(label='Начало утреннего максимума',
                                              options=list_t, index=17), '%H:%M')
            time_start_morning = select_dtm_start.hour + select_dtm_start.minute / 60

            ###### время в часах конца утреннего максимума
            select_dtm_finish = datetime.datetime. \
                strptime(st.sidebar.selectbox(label='Конец утреннего максимума',
                                              options=list_t, index=22), '%H:%M')
            time_finish_morning = select_dtm_finish.hour + select_dtm_finish.minute / 60

            st.sidebar.markdown('#### Вечерний максимум')

            # Список времени в в строках 00:00, 00:30... str
            list_t = list_times(hour_start=15, hour_finish=24, minute_delta=30)

            # время в часах начала вечернего максимума
            select_dtm_start = datetime.datetime. \
                strptime(st.sidebar.selectbox(label='Начало вечернего максимума',
                                              options=list_t, index=7), '%H:%M')
            time_start_evening = select_dtm_start.hour + select_dtm_start.minute / 60

            # время в часах конца утреннего максимума
            select_dtm_finish = datetime.datetime. \
                strptime(st.sidebar.selectbox(label='Конец вечернего максимума',
                                              options=list_t, index=12), '%H:%M')
            time_finish_evening = select_dtm_finish.hour + select_dtm_finish.minute / 60


        agree_limit = st.sidebar.checkbox('Расчет лимитов мощности')
        if agree_limit:
            st.subheader('Определение математически ожидаемых и максимальных значений получасовой мощности ')
            st.write('Часы утреннего и вечернего максимумов нагрузки энергосистемы и периоды их контроля, '
                     'но не более шести часов в сутки, устанавливает энергоснабжающая организация и доводит их до '
                     'абонентов в письменной форме с последующим отражением в договоре электроснабжения.')
            st.write('Максимальная мощность определяется наибольшим значением из ряда мощностей, '
                     'расположенных в пиковой зоне, при этом значение максимальной мощности определяется '
                     'как верхний предел доверительного интервала для среднего значения.')

            st.markdown('##### Моделируемая граница максимальной мощности')
            df1 = df_power_statistics
            if time_start_morning and time_finish_morning is not None:
                power_limits = an.PowerLimits(df=df1, time_start_morning=time_start_morning,
                                              time_finish_morning=time_finish_morning,
                                              time_start_evening=time_start_evening,
                                              time_finish_evening=time_finish_evening)
            else:
                power_limits = an.PowerLimits(df=df1)

            power_limits.power_limits()
            df_only_max_period = power_limits.df_only_max_period
            df_only_max_period = df_only_max_period.iloc[:, [0, 4, 5, 6, 7, 8, 9, 10]]
            st.write(df_only_max_period.astype(str))

            st.markdown('##### Диаграмма средней, максимальной и моделируемой предельной мощности с 5-и % вероятностью '
                        'возникновения усреднённая суточными наблюдениями ')
            df_full_limit = power_limits.power_limits()
            df_full_limit = df_full_limit.rename(columns={'Month': 'Месяц'})
            drop_list = ['Год', 'Месяц', 'Час', 'Часы суток',
                         'Минимальная активная мощность, кВт',
                         'Среднеквадратическое отклонение активной мощности, кВт',
                         'Количество значений', 'Период наблюдений' ]
            df_full_limit = df_full_limit.drop(drop_list, axis=1)

            st.line_chart(data=df_full_limit, width=0, height=0, use_container_width=True)

            st.markdown('##### Ожидаемые лимиты мощности следующего за анализируемым периодом')
            df_max_month_value = power_limits.df_max_month_value.reset_index()
            df_max_month_value = df_max_month_value.iloc[:, [1, 2, 3]]
            st.write(df_max_month_value.astype(str))

            st.markdown('##### Диаграмма прогнозируемых лимитов мощности следующего за анализируемым периодом')
            df_max_month_value = power_limits.df_max_month_value
            dfx = df_max_month_value.iloc[:, [0]].astype(str)
            dfy_1= df_max_month_value.iloc[:, [1]].astype(float)
            dfy_2 = df_max_month_value.iloc[:, [2]].astype(float)
            fig = mp.my_histogram(dfx, dfy_1, dfy_2, )
            st.write(fig)

            # Запись всех результатов Д-тарифа с заявленной мощностью
            write_to_exel_buffer = power_limits.write_to_exel_buffer()
            st.download_button(label='Сохранить результаты расчета лимитов мощности в xlsx файл',
                               data=write_to_exel_buffer,
                               file_name='Результаты расчета лимитов мощности.xlsx')

            st.write('_________________________')

        st.sidebar.write('____')



        st.sidebar.markdown('#### Анализ стоимости электроэнергии')
        check_analysis_tariff = st.sidebar.checkbox('Расчет стоимости электроэнергии')
        # Делаем предварительные расчеты

        if check_analysis_tariff:
            st.subheader('Анализ стоимости электроэнергии при различных тарифах оплаты за электроэнергию')
            df1 = df_power_statistics
            df3 = df_declared
            # Базовый курс доллара США
            kb = float(df_initial_data.iloc[2, 1])
            # Текущий курс доллара США
            kt = float(df_initial_data.iloc[3, 1])
            # Основная плата – за мощность (на 1 месяц)
            ab = float(df_initial_data.iloc[4, 1])
            # Дополнительная плата – за энергию
            bb = float(df_initial_data.iloc[5, 1])

            if time_start_morning and time_finish_morning is not None:
                d_tariff = an.DTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb,
                                      time_start_morning=time_start_morning, time_finish_morning=time_finish_morning,
                                      time_start_evening=time_start_evening, time_finish_evening=time_finish_evening)
                d_tariff_declared = an.DTariffDeclared(df=df1, ab=ab, bb=bb, kt=kt, kb=kb, declared=df3)
            else:
                d_tariff = an.DTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
                d_tariff_declared = an.DTariffDeclared(df=df1, ab=ab, bb=bb, kt=kt, kb=kb, declared=df3)




            # dif_tariff = analytic.DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
            # dif_tariff_reg = analytic.DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)



            check_declared = st.checkbox('Двухставочный тариф за с оплатой за заявленную мощность')
            if check_declared:
                st.subheader('Двухставочный тариф с оплатой за заявленную (договорную) мощность')
                st.write('При применении двухставочного тарифа с основной ставкой а (руб/кВт) за 1 кВт заявленной '
                         'договорной величины наибольшей потребляемой активной мощности в часы утреннего и '
                         'вечернего пиков энергосистемы (Pз.max) и дополнительной ставкой b (руб/кВт∙ч) за 1 кВт∙ч '
                         'потребляемой активной энергии  за расчетный месяц суммарная плата за электрическую '
                         'энергию определяется по формуле:')
                st.latex(r'''П_{д.з} = {a} \cdotp {P_{з.max}} + {b} \cdotp {W}''')
                st.write('_________________________')
                st.markdown('##### Анализ электропотребления за расчетный период исследования')
                energy_analyzer = d_tariff.energy_analyzer_month().reset_index()
                st.write(energy_analyzer.astype(str))

                dfx = energy_analyzer.iloc[:, [0]].astype(str)
                dfy_full_energy = energy_analyzer.iloc[:, [4]].astype(float)
                fig = mp.my_histogram(dfx, dfy_full_energy)
                st.markdown('##### Изменение электропотребления')
                st.write(fig)

                st.write('_________________________')

                st.markdown('##### Результаты расчета оплаты за электроэнергию при оплате по двухставочному '
                            'тарифу с заявленной мощностью')
                d_tariff_declared.calculation()
                d_tariff_decl_for_table = d_tariff_declared.df_pay_energy_month.reset_index()

                st.write(d_tariff_decl_for_table.astype(str))

                sum_pay_power_and_energy = d_tariff_declared.sum_pay_power_and_energy

                # Расход электроэнергии
                sum_energy = d_tariff.sum_energy
                # Суммарная плата за мощность
                sum_pay_power = d_tariff_declared.sum_pay_power
                # Суммарная плата за ЭЭ
                sum_pay_energy = d_tariff_declared.sum_pay_energy
                # процент оплаты за мощность в суммарной оплате
                dec_per = round(sum_pay_power / sum_pay_power_and_energy * 100, 1)


                st.write('_________________________')

                st.markdown('##### Изменение оплаты за электрическую мощность и энергию')

                dt_bar = d_tariff_declared.df_pay_energy_month.reset_index().iloc[:, [0, 5, 8, 9]]
                dfx = dt_bar.iloc[:, [0]].astype(str)
                dfy_pay_power = dt_bar.iloc[:, [1]].astype(float)
                dfy_pay_energy = dt_bar.iloc[:, [2]].astype(float)
                dfy_pay_power_energy = dt_bar.iloc[:, [3]].astype(float)

                fig = mp.my_histogram(dfx, dfy_pay_power, dfy_pay_energy, dfy_pay_power_energy, y_scale_min=0)
                st.write(fig)

                st.write('_________________________')
                st.markdown('##### Круговая диаграмма суммарной оплаты за заявленную электрическую '
                            'мощность и энергию')
                list_pay_dec = [sum_pay_energy, sum_pay_power]
                list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']

                fig = mp.my_pie(list_values=list_pay_dec, list_labels=list_labels)
                st.write(fig)

                st.write('_________________________')
                st.subheader(f'Наиболее значимые результаты')
                st.metric(label="Суммарный расход электроэнергии в рассмотренном периоде", value=f'{sum_energy} кВт∙ч.')
                st.metric(label="Суммарная оплата за мощность в рассмотренном периоде", value=f'{sum_pay_power} руб.')
                st.metric(label="Суммарная оплата за электроэнергию в рассмотренном периоде",
                          value=f'{sum_pay_energy} руб.')
                st.metric(label="Итоговая оплата за мощность и электроэнергию в рассмотренном периоде",
                          value=f'{sum_pay_power_and_energy} руб.')
                st.metric(label="Доля оплаты за мощность в общих денежных затратах составляет",
                          value=f'{dec_per} %')

                # Запись всех результатов Д-тарифа с заявленной мощностью
                write_to_exel_buffer = d_tariff_declared.write_to_exel_buffer()
                st.download_button(label='Сохранить результаты анализа Д-тарифа с заявленной мощностью в xlsx файл',
                                   data=write_to_exel_buffer,
                                   file_name='Результаты анализа Д-тарифа с заявленной мощностью.xlsx')

                st.write('_________________________')

            check_d_tariff = st.checkbox('Двухставочный тариф за с оплатой за фактическую мощность')
            if check_d_tariff:
                st.subheader('Двухставочный тариф с оплатой за фактическую мощность')
                st.write('При использовании двухставочного тарифа с основной ставкой а за 1 кВт фактической величины'
                         ' наибольшей потребляемой активной мощности в часы утреннего и вечернего пиков энергосистемы'
                         ' (Рф.max) и дополнительной ставкой b (руб/кВт∙ч) за 1 кВт∙ч потребляемой активной '
                         'энергии за расчетный месяц суммарная плата за электрическую энергию определяется по формуле:')
                st.latex(r'''П_{д.ф} = {a} \cdotp {P_{ф.max}} + {b} \cdotp {W}''')
                st.write('_________________________')
                st.markdown('##### Анализ получасовых максимумов нагрузок')
                st.write('Плата основной ставки системы двухставочного тарифа осуществляется за наибольшую величину'
                         ' потребляемой активной мощности в часы утреннего и вечернего пиков энергосистемы в расчетном'
                         ' периоде. В связи с чем экономически целесообразным регулированием для предприятия является'
                         ' максимально-возможное снижение активной мощности в период максимума энергосистемы.'
                         ' Наибольший интерес представляет границы нахождения наибольшего максимума активной'
                         ' мощности, что определяет сосредоточенность существующей электрической нагрузки')

                power_analyzer_month = d_tariff.power_analyzer_month().reset_index()
                st.write(power_analyzer_month.astype(str))
                st.info('Индикатор в таблице соответствует: \n'
                        'True - в случае, если максимум активной мощности выходит '
                        'за границы максимумов энергосистемы; False - в обратном случае.')

                # Посуточный анализ данных
                power_analyzer_day = d_tariff.power_analyzer_day().reset_index()
                # Количество суток исследования
                count_day = power_analyzer_day['Индикатор'].count()
                # Количество суток где наибольший максимум вышел за границы максимума энергосистемы
                count_day_true = power_analyzer_day['Индикатор'].sum()
                # Вывод
                val = round(count_day_true / count_day * 100, )
                if val > 50:
                    conclusion1 = f'Регулирование 30-и минутных максимумов нагрузок можно считать эффективным, ' \
                                  f'поскольку преимущественно  ({val} % случаев) ' \
                                  f'максимальная нагрузка выходит за границы максимума энергосистемы'
                else:
                    conclusion1 = f'Регулирование 30-и минутных максимумов максимумов нагрузок является низкоэффективным, ' \
                                  f'поскольку преимущественно  ({val} % случаев) ' \
                                  f'максимальная нагрузка находится в границах максимума энергосистемы'

                st.subheader(f'Заключение анализа максимумов нагрузок')
                st.write(f'В результате оценки максимумов установлено, что {count_day_true} из {count_day} дней выходят '
                         f'за границы максимумов энергосистемы, следовательно:')
                st.markdown(f'######  {conclusion1}.')

                # Сравнение максимумов утро, вечер и максимум, котоырй наблюдался
                power = power_analyzer_month
                dfx = power.iloc[:, [0]].astype(str)
                dfy_1 = power.iloc[:, [1]].astype(float)
                dfy_2 = power.iloc[:, [2]].astype(float)
                dfy_3 = power.iloc[:, [3]].astype(float)
                fig = mp.my_histogram(dfx, dfy_1, dfy_2, dfy_3)
                st.write(fig)


                # Сравнение мощностей по суткам
                st.markdown('##### Сравнение максимумов нагрузок по суткам')
                power_analyzer_day = d_tariff.power_analyzer_day().reset_index()
                st.write(power_analyzer_day.astype(str))
                st.write('___')

                st.markdown('##### Результаты расчета электропотребления за расчетный период исследования')
                energy_analyzer = d_tariff.energy_analyzer_month().reset_index()
                st.write(energy_analyzer.astype(str))

                dfx = energy_analyzer.iloc[:, [0]].astype(str)
                dfy_full_energy = energy_analyzer.iloc[:, [4]].astype(float)
                fig = mp.my_histogram(dfx, dfy_full_energy)
                st.markdown('##### Изменение электропотребления')
                st.write(fig)
                st.write('_________________________')

                st.markdown('##### Результаты расчета оплаты за электроэнергию при оплате по двухставочному '
                            'тарифу с заявленной мощностью')
                d_tariff.calculation()
                pay_month = d_tariff.df_pay_energy_month.reset_index()
                st.write(pay_month.astype(str))
                st.write('_________________________')

                st.markdown('##### Изменение оплаты за электрическую мощность и энергию')
                dt_bar = pay_month.iloc[:, [0, 5, 8, 9]]
                dfx = dt_bar.iloc[:, [0]].astype(str)
                dfy_pay_power = dt_bar.iloc[:, [1]].astype(float)
                dfy_pay_energy = dt_bar.iloc[:, [2]].astype(float)
                dfy_pay_power_energy = dt_bar.iloc[:, [3]].astype(float)

                fig = mp.my_histogram(dfx, dfy_pay_power, dfy_pay_energy, dfy_pay_power_energy, y_scale_min=0)
                st.write(fig)

                st.write('_________________________')
                st.markdown('##### Круговая диаграмма суммарной оплаты за заявленную электрическую '
                            'мощность и энергию')
                sum_pay_energy = d_tariff.sum_pay_energy
                sum_pay_power = d_tariff.sum_pay_power

                list_pay_dec = [sum_pay_energy, sum_pay_power]
                list_labels = ['Cуммарная плата за электроэнергию', 'Cуммарная плата за мощность']

                fig = mp.my_pie(list_values=list_pay_dec, list_labels=list_labels)
                st.write(fig)
                st.write('_________________________')

                st.subheader(f'Наиболее значимые результаты')
                # Расход электроэнергии
                sum_energy = d_tariff.sum_energy
                # Суммарная плата за мощность
                sum_pay_power = d_tariff.sum_pay_power
                # Суммарная плата за ЭЭ
                sum_pay_energy = d_tariff.sum_pay_energy
                # Оплата за ЭЭ итого
                sum_pay_power_and_energy = d_tariff.sum_pay_power_and_energy
                # процент оплаты за мощность в суммарной оплате
                dec_per = round(sum_pay_power / sum_pay_power_and_energy * 100, 1)

                st.metric(label="Суммарный расход электроэнергии в рассмотренном периоде", value=f'{sum_energy} кВт∙ч.')
                st.metric(label="Суммарная оплата за мощность в рассмотренном периоде", value=f'{sum_pay_power} руб.')
                st.metric(label="Суммарная оплата за электроэнергию в рассмотренном периоде",
                          value=f'{sum_pay_energy} руб.')
                st.metric(label="Итоговая оплата за мощность и электроэнергию в рассмотренном периоде",
                          value=f'{sum_pay_power_and_energy} руб.')
                st.metric(label="Доля оплаты за мощность в общих денежных затратах составляет",
                          value=f'{dec_per} %')

                # Запись всех результатов Д-тарифа с заявленной мощностью
                write_to_exel_buffer = d_tariff.write_to_exel_buffer()
                st.download_button(label='Сохранить результаты анализа Д-тарифа с фактической мощностью в xlsx файл',
                                   data=write_to_exel_buffer,
                                   file_name='Результаты анализа Д-тарифа с фактической мощностью.xlsx')
                st.write('_________________________')

            check_d_tariff = st.checkbox('Двухставочно-дифференцированный тариф без изменения режима работы предприятия')
            if check_d_tariff:
              pass









    except Exception as e:
        st.text(traceback.format_exc())
        pass


def app_authorization():
    st.title('Калькулятор тарифов')
    st.info('Определение оптимального тарифа оплаты за электроэнергию  '
            'позволяет экономить от 5 до 20 % от стоимости ресурсов ежегодно.')

    st.sidebar.markdown('# Авторизация')
    menu = ['Авторизация пользователя', 'Демо режим']
    menu_choice = st.sidebar.selectbox('Режим входа', menu)
    st.sidebar.write('________')
    if menu_choice == 'Авторизация пользователя':
        st.sidebar.write('Для работы с ресурсом введите данные авторизации')
        user_name = st.sidebar.text_input('Имя пользователя')
        password = st.sidebar.text_input('Пароль', type="password")


        if user_name == 'root' and password == '123':
            st.sidebar.success('Вы вошли под учетной записью администратора')
            with st.expander('ВВОД НОВЫХ ПОЛЬЗОВАТЕЛЕЙ'):
                st.markdown('##### Добавление нового пользователя в базу данных')
                col1, col2 = st.columns(2)
                with col1:
                    new_user_name = st.text_input('Имя нового пользователя')
                with col2:
                    new_password = st.text_input('Пароль для нового пользователя', type="password")

                year_now = datetime.datetime.now().year
                month_now = datetime.datetime.now().month
                date_finish_servise = st.date_input('Дата окончания подписки',  value=datetime.date(year_now+1,
                                                                                                    month_now+2, 1))
                if st.button('Ввести'):
                    create_user_table()
                    add_user_data(username=new_user_name, password=new_password,
                                  date_finish_servise=date_finish_servise)
                st.write('_________')

                st.markdown('###### Удаление пользователей из база данных')
                delete_user_name = st.text_input('Имя пользователя для удаления')
                if st.button('Удалить'):
                    delete_user_from_bd(username=delete_user_name)

                st.markdown('###### Существующая база данных')
                result_user = view_all_user()
                st.write(result_user)
            app()
        elif user_name == '' and password == '':
            pass
        elif check_user(username=user_name, password=password) == 1:
            st.sidebar.success('Вы успешно вошли в приложение')
            st.sidebar.success(f'Активно до: { data_activate(username=user_name, password=password)}')
            st.sidebar.write('_________')
            app()

        elif check_user(username=user_name, password=password) == 0:
            st.error('Пользователь с веденной учетной записью отсутствует')
            st.sidebar.write('_________')
            app()

    else:
        app(mode='demo')
        pass



# Создание базы данных для входа пользователей
# https://www.youtube.com/watch?v=HU_kd-1uIkQ
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_user_table():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT UNIQUE, password TEXT NOT NULL, '
              'date_finish_servise DATE)')


def add_user_data(username, password, date_finish_servise):
    c.execute('INSERT INTO usertable(username, password, date_finish_servise) VALUES (?, ?, ?) '
              'ON CONFLICT(username) DO NOTHING', # Если имя повторяется - ничего не делает
              (username, password, date_finish_servise))
    conn.commit()

def check_user(username, password):
    c.execute('SELECT * FROM usertable WHERE username =? and password=?', (username, password))
    len_data=len(c.fetchall())
    if len_data==0:
        return 0
    else:
        return 1

def data_activate(username, password):
    data = c.execute('SELECT date_finish_servise FROM usertable WHERE (username =?) and (password=?)',
                     [username, password]).fetchall()[0][0]
    return data

def delete_user_from_bd(username):
    c.execute('DELETE FROM usertable WHERE username=?', [username])
    conn.commit()

def view_all_user():
    c.execute('SELECT * FROM usertable')
    df = pd.read_sql("SELECT * FROM usertable ", conn)
    return df

if __name__ == '__main__':
    app_authorization()

conn.close()
