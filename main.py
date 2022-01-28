import streamlit as st
import pandas as pd
from package_power_analytics.template import create_template_input_data, write_to_excel
import package_power_analytics.myplotly as mp
import package_power_analytics.analytic as an
import os, sys
import datetime
import traceback
from io import BytesIO
import base64
import numpy as np
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


def app():
    st.markdown('### АНАЛИЗАТОР ЭЛЕКТРИЧЕСКОЙ НАГРУЗКИ')
    try:
        with st.expander('ГЕНЕРАЦИЯ ШАБЛОНА ДЛЯ ВВОДА ИСХОДНЫХ ДАННЫХ ПРОЕКТА'):
            year_now = datetime.datetime.now().year
            year_last = year_now - 1
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.date_input('Начальная дата', value=datetime.date(year_last, 1, 1))
            with col2:
                finish_time = st.date_input('Конечная дата', value=datetime.date(year_last, 12, 31))

            file_xlsx = create_template_input_data(start_time, finish_time)
            st.download_button(label='Сгенерировать шаблон для ввода данных Excel-файла',
                                       data=file_xlsx,
                                       file_name='Шаблон ввода исходных данных.xlsx')

        with st.expander('ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ ПРОЕКТА'):

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
                agree = st.checkbox('Показать загруженные данные')
                if agree:
                    st.markdown('##### Исходные данные')

                    for row_index,row in df_initial_data.iterrows():
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

                df_power_statistics = xls.parse('Получасовая статистика')
                if df_power_statistics.isnull().values.any() == True:
                    st.markdown('**Получасовая статистика активной и реактивной мощности не должна'
                                'содержать пропусков!**')
                    raise Exception ('Ошибка. Не заполнены данные в исходной статистике.')

                # Ручное редактирование
                st.sidebar.header('Настройки расчета')
                st.sidebar.markdown('#### Редактирование исходных данных')
                agree = st.sidebar.checkbox('Включить ручное редактирование исходных данных')
                if agree:
                    df_initial_data.iloc[0,1] = st.sidebar.text_input(label='Название предприятия',
                                                                      value=df_initial_data.iloc[0,1])
                    df_initial_data.iloc[2, 1] = st.sidebar.text_input(label='Базовый курс доллара США, Кб, руб/долл. США',
                                                                       value=float(df_initial_data.iloc[2, 1]))
                    df_initial_data.iloc[3, 1] = st.sidebar.text_input(label='Текущий курс доллара, Кт, руб/долл. США',
                                                                       value=float(df_initial_data.iloc[3, 1]))
                    df_initial_data.iloc[4, 1] = st.sidebar.text_input(label='Основная плата - за мощность, а, руб/(кВт*мес)',
                                          value=float(df_initial_data.iloc[4, 1]))
                    df_initial_data.iloc[5, 1] = st.sidebar.text_input(label='Дополнительная плата - за энергию , b, руб/кВтч',
                                          value=float(df_initial_data.iloc[5, 1]))
                    option =  st.sidebar.selectbox(label='Существующий тариф оплаты',
                                                   options = ('Д-тариф с заявленной мощностью',
                                                    'Д-тариф с фактической мощностью',
                                                    'ДД-тариф'), index = df_initial_data.iloc[8, 1])
                    if option == 'Д-тариф с заявленной мощностью':
                        df_initial_data.iloc[8, 1] = float(0)
                    elif option == 'Д-тариф с фактической мощностью':
                        df_initial_data.iloc[8, 1] = float(1)
                    elif option == 'ДД-тариф':
                        df_initial_data.iloc[8, 1] = float(2)

                    file_xlsx_new = create_template_input_data(start_time=None, finish_time=None, df1=df_power_statistics,
                                                           df2=df_declared, df3=df_initial_data)
                    st.sidebar.download_button(label='Сохранить изменения в Excel-файл',
                                       data=file_xlsx_new, file_name=f'Исходные данные {df_initial_data.iloc[0,1]}.xlsx')

        st.sidebar.markdown('#### Предварительный анализ данных')
        agree_describe = st.sidebar.checkbox('Выполнить предварительный анализ')
        if agree_describe:
            with st.expander('ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ДАННЫХ'):
                st.markdown('##### Предварительный анализ данных')
                agree_input_data = st.checkbox('Показать исходные данные')
                if agree_input_data:
                    st.write(df_power_statistics)

                show_describe = st.checkbox('Показать описательную статистику')
                if show_describe:
                    st.write('Описательная статистика позволяет обобщать первичные результаты, '
                             'полученные при наблюдении или в эксперименте. В качестве статистических показателей '
                             'используются: среднее, медиана, мода, дисперсия, стандартное отклонение и др.')
                    res1 = an.describe_statistics(one_d_df=df_power_statistics.iloc[:, [1]])
                    st.markdown(f'###### {df_power_statistics.iloc[:, [1]].columns[0]}')
                    st.write(res1)

                    res2= an.describe_statistics(one_d_df=df_power_statistics.iloc[:, [2]])
                    st.markdown(f'###### {df_power_statistics.iloc[:, [2]].columns[0]}')
                    st.write(res2)
                    #Запись результатов в файл
                    file_xlsx = write_to_excel(res1, res2)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                      data=file_xlsx,
                                      file_name='Результаты описательной статистики.xlsx')

                show_activ_power = st.checkbox('Показать график изменения активной мощности')
                if show_activ_power:
                    dfx = df_power_statistics.iloc[:,[0]]
                    dfy_act = df_power_statistics.iloc[:, [1]]
                    fig = mp.myplotly(dfx, dfy_act)
                    st.write(fig)
                show_reactiv_power = st.checkbox('Показать график изменения реактивной мощности')
                if show_reactiv_power:
                    dfx = df_power_statistics.iloc[:,[0]]
                    dfy_react = df_power_statistics.iloc[:, [2]]
                    fig = mp.myplotly(dfx, dfy_react)
                    st.write(fig)

                power_coefficients_fi = df_power_statistics
                power_coefficients_fi['Полная мощность, кВА'] = \
                    round(np.sqrt(df_power_statistics.iloc[:, 1] ** 2 + df_power_statistics.iloc[:, 2] ** 2))
                power_coefficients_fi['Коэффициент активной мощности (cosф)'] = df_power_statistics.iloc[:,1] / \
                                                                                df_power_statistics.iloc[:,3]
                show_coefficients_fi = st.checkbox('Показать график изменения коэффициента активной мощности')
                if show_coefficients_fi:
                    dfx = power_coefficients_fi.iloc[:,[0]]
                    dfy_react = power_coefficients_fi.iloc[:, [4]]
                    fig = mp.myplotly(dfx, dfy_react)
                    st.write(fig)

        st.sidebar.markdown('#### Анализ графиков нагрузок в разрезе месяца')
        check_coefficients = st.sidebar.checkbox('Выполнить анализ графиков нагрузок')
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
            with st.expander('АНАЛИЗ ГРАФИКОВ НАГРУЗОК В РАЗРЕЗЕ МЕСЯЦА'):
                st.markdown('##### Определение основных физических величин графиков нагрузки')
                check_mean_power = st.checkbox('Расчет средней месячной нагрузки')
                if check_mean_power:
                    st.markdown('###### Средняя нагрузка')
                    st.write('Средняя нагрузка – это постоянная, неизменная величина за любой рассматриваемый '
                             'промежуток времени, которая вызывает такой же расход электроэнергии, '
                             'как и изменяющаяся за это время нагрузка.')
                    st.latex(r'''P_с =  \frac {\sum_{i=1}^{n} P_{сi} }  {N}''')

                    df_mean_str = df_mean.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_mean_str)

                    dfx = df_mean_str.iloc[:, [0]]
                    dfy_mean_act = df_mean_str.iloc[:, [1]]
                    dfy_mean_react = df_mean_str.iloc[:, [2]]
                    dfy_mean_full = df_mean_str.iloc[:, [3]]
                    fig = mp.my_histogram(dfx, dfy_mean_act, dfy_mean_react, dfy_mean_full, )
                    st.write(fig)

                    act_mean = round(df_mean.iloc[:, 1].mean(), 2)
                    react_mean = round(df_mean.iloc[:, 2].mean(), 2)
                    full_mean = round(df_mean.iloc[:, 3].mean(), 2)

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - средняя активная нагрузка в исследуемом периоде: **{act_mean}** кВт')
                    st.markdown(f'###### - средняя реактивная нагрузка в исследуемом периоде: **{react_mean}** кВАр')
                    st.markdown(f'###### - средняя полная нагрузка в исследуемом периоде: **{full_mean}** кВА')

                    #Запись результатов в файл
                    file_xlsx = write_to_excel(df_mean)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                      data=file_xlsx,
                                      file_name='Результаты расчета средней нагрузки.xlsx')

                # Расчет среднеквадратичной нагрузки
                check_square_power = st.checkbox('Расчет среднеквадратичной месячной нагрузки')
                if check_square_power:
                    st.markdown('###### Среднеквадратичная месячная нагрузка')
                    st.write('Среднеквадратичная нагрузка – это постоянная, неизменная нагрузка за любой рассматриваемый'
                             ' промежуток времени, которая обуславливает такие же потери мощности в проводниках, '
                             'как и изменяющаяся за это время нагрузка.')
                    st.latex(r'''P_{ск} = \sqrt { \frac  {\sum_{i=1}^{n} P^{2}_{сi} }  {N} }''')

                    df_square_str = df_square.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_square)

                    dfx = df_square_str.iloc[:, [0]]
                    dfy_square_act = df_square_str.iloc[:, [1]]
                    dfy_square_react = df_square_str.iloc[:, [2]]
                    dfy_square_full = df_square_str.iloc[:, [3]]
                    fig = mp.my_histogram(dfx, dfy_square_act, dfy_square_react, dfy_square_full )
                    st.write(fig)

                    act_square = round(df_square.iloc[:, 1].mean(), 2)
                    react_square = round(df_square.iloc[:, 2].mean(), 2)
                    full_square = round(df_square.iloc[:, 3].mean(), 2)

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - средневадратичная активная нагрузка в '
                                f'исследуемом периоде: **{act_square}** кВт')
                    st.markdown(f'###### - средневадратичная реактивная нагрузка в '
                                f'исследуемом периоде: **{react_square}** кВАр')
                    st.markdown(f'###### - средневадратичная полная нагрузка в '
                                f'исследуемом периоде: **{full_square}** кВА')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_square)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета среднеквадратичной нагрузки.xlsx')

                # Расчет максимальной месячной нагрузки
                check_max_power = st.checkbox('Расчет максимальной месячной нагрузки')
                if check_max_power:
                    st.markdown('###### Максимальная месячная нагрузка')
                    st.write(
                        'Максимальная нагрузка – это наибольшая из средних нагрузок за '
                        'рассматриваемый промежуток времени.')
                    st.latex(r'''P_{max} = max(P_i)''')

                    df_max_str = df_max.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_max_str)

                    dfx = df_max_str.iloc[:, [0]]
                    dfy_max_act = df_max_str.iloc[:, [1]]
                    dfy_max_react = df_max_str.iloc[:, [2]]
                    dfy_max_full = df_max_str.iloc[:, [3]]
                    fig = mp.my_histogram(dfx, dfy_max_act, dfy_max_react, dfy_max_full)
                    st.write(fig)

                    act_max = round(df_max.iloc[:, 1].max(), 2)
                    react_max = round(df_max.iloc[:, 2].max(), 2)
                    full_max = round(df_max.iloc[:, 3].max(), 2)

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - средневадратичная активная нагрузка в '
                                f'исследуемом периоде: **{act_max}** кВт')
                    st.markdown(f'###### - средневадратичная реактивная нагрузка в '
                                f'исследуемом периоде: **{react_max}** кВАр')
                    st.markdown(f'###### - средневадратичная полная нагрузка в '
                                f'исследуемом периоде: **{full_max}** кВА')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_max)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета максимальной нагрузки.xlsx')

                st.markdown('##### Определение безразмерных физических величин графиков нагрузки')

               # Коэффициент максимума график
                check_cof_max_power = st.checkbox('Расчет коэффициента максимума графика')
                if check_cof_max_power:
                    st.markdown('###### Коэффициент максимума график')
                    st.write(
                        'Наряду с физическими величинами графики нагрузки описываются безразмерными коэффициентами. '
                        'Эти коэффициенты устанавливают связь между основными физическими величинами, характеризуют '
                        'неравномерность графиков нагрузки, а также использование электроприёмников и потребителей '
                        'электроэнергии по мощности и времени.')

                    st.latex(r'''K_{м.г} = \frac  {P_м}  {P_{ср}} ''')

                    df_coef_max_str = df_coef_max.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_coef_max_str)

                    dfx = df_coef_max_str.iloc[:, [0]]
                    dfy_coef_max = df_coef_max_str.iloc[:, [1]]
                    fig = mp.my_histogram(dfx, dfy_coef_max)
                    st.write(fig)

                    max_coef = round(df_coef_max.iloc[:, 1].max(), 2)
                    df_temp = df_coef_max.loc[df_coef_max.iloc[:, 1] == df_coef_max.iloc[:, 1].max()]
                    max_mohtn_coef = df_temp.iloc[:, 0].values[0]

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - наибольшее значение коэффициента максимума в '
                                f'исследуемом периоде: **{max_coef}**')
                    st.markdown(f'###### - месяц с наибольшим коэффициентом максимума '
                                f' **{max_mohtn_coef}**')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_coef_max)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета коэффициента максимума.xlsx')

                # Коэффициент заполенния графика
                check_coef_fill_power = st.checkbox('Расчет коэффициента заполнения графика')
                if check_coef_fill_power:
                    st.markdown('###### Коэффициент заполнения графика.')
                    st.write('Характеризует неравномерность графика нагрузок. Для рациональной передачи электроэнергии, '
                                'т.е. снижения к минимуму потерь в питающих и распределительных сетях предприятия, '
                                'коэффициент заполнения графика должен быть приближен к единице, '
                                'что определяется равномерностью включения технологических потребителей электроэнергии')

                    st.latex(r'''K_{з.г} = \frac  {P_{ср}}  {P_{м}} ''')

                    df_coef_fill_str = df_coef_fill.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_coef_fill_str)

                    dfx = df_coef_fill_str.iloc[:, [0]]
                    dfy_coef_fill = df_coef_fill_str.iloc[:, [1]]
                    fig = mp.my_histogram(dfx, dfy_coef_fill)
                    st.write(fig)

                    max_coef_fill = round(df_coef_fill.iloc[:, 1].max(), 2)
                    df_temp = df_coef_fill.loc[df_coef_fill.iloc[:, 1] == df_coef_fill.iloc[:, 1].max()]
                    max_mohtn_coef = df_temp.iloc[:, 0].values[0]


                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - наибольшее значение коэффициента заполнения в '
                                f'исследуемом периоде: **{max_coef_fill}**')
                    st.markdown(f'###### - месяц с наибольшим коэффициентом заполнения '
                                f' **{max_mohtn_coef}**')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_coef_fill)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета коэффициента заполнения.xlsx')

                # Коэффициент формы графика
                check_shape_power = st.checkbox('Расчет коэффициента формы графика')
                if check_shape_power:
                    st.markdown('###### Коэффициент формы графика')
                    st.write('Характеризует неравномерность график нагрузок. Для минимизации потерь электроэнергии '
                             'в питающих, распределительных сетях и трансформаторах системы существующей системы '
                             'электроснабжения коэффициент формы графика должен быть приближен к единице.')

                    st.latex(r'''K_{ф.г} = \frac  {P_{ск}}  {P_{с}} ''')


                    df_coef_shape_str = df_coef_shape.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_coef_shape_str)

                    dfx = df_coef_shape_str.iloc[:, [0]]
                    dfy_coef_fill = df_coef_shape_str.iloc[:, [1]]
                    fig = mp.my_histogram(dfx, dfy_coef_fill)
                    st.write(fig)

                    max_coef_shape = round(df_coef_shape.iloc[:, 1].max(), 2)
                    df_temp = df_coef_shape.loc[df_coef_shape.iloc[:, 1] == df_coef_shape.iloc[:, 1].max()]
                    max_mohtn_shape= df_temp.iloc[:, 0].values[0]

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - наибольшее значение коэффициента заполнения в '
                                f'исследуемом периоде: **{max_coef_shape}**')
                    st.markdown(f'###### - месяц с наибольшим коэффициентом заполнения '
                                f' **{max_mohtn_shape}**')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_coef_shape)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета коэффициента формы.xlsx')

                # Коэффициент активной мощности
                check_coefficient_fi = st.checkbox('Расчет коэффициента активной мощности')
                if check_coefficient_fi:
                    st.markdown('###### Коэффициент активной мощности')
                    st.write('Коэффициент активной мощности – безразмерная физическая величина, '
                             'характеризующая потребителя переменного электрического тока с точки зрения '
                             'наличия в нагрузке реактивной составляющей.')

                    st.latex(r'''\cos {\phi} = \frac  {P_{ср}}  {S_{ср}} ''')

                    df_coef_fi_str = df_coef_fi.astype(str)
                    st.write('Результаты расчета')
                    st.write(df_coef_fi_str)

                    dfx = df_coef_fi_str.iloc[:, [0]]
                    dfy_coef_fi = df_coef_fi_str.iloc[:, [1]]
                    fig = mp.my_histogram(dfx, dfy_coef_fi)
                    st.write(fig)

                    max_coef_fi = round(df_coef_fi.iloc[:, 1].max(), 3)
                    min_coef_fi = round(df_coef_fi.iloc[:, 1].min(), 3)

                    st.markdown('##### В результаты анализа данных установлено:')
                    st.markdown(f'###### - наибольшее значение коэффициента мощности в '
                                f'исследуемом периоде: **{max_coef_fi}**')
                    st.markdown(f'###### - наименьшее значение коэффициента мощности в '
                                f'исследуемом периоде: **{min_coef_fi}**')

                    # Запись результатов в файл
                    file_xlsx = write_to_excel(df_coef_fi)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                       data=file_xlsx,
                                       file_name='Результаты расчета коэффициента мощности.xlsx')

                # Запись всех результатов анализа графика в файл
                file_xlsx = write_to_excel(df_mean, df_square, df_max, df_coef_max, df_coef_fill,
                                           df_coef_shape, df_coef_fi)
                st.download_button(label='Сохранить групповые результаты в xlsx файл',
                                   data=file_xlsx,
                                   file_name='Результаты расчета показателей ГЭН.xlsx')
















    except Exception as e:
        st.text(traceback.format_exc())






if __name__ == '__main__':
    app()