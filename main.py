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

        st.sidebar.markdown('#### Анализ графиков нагрузок')
        check_coefficients = st.sidebar.checkbox('Выполнить анализ графиков нагрузок')
        power_coefficients = an.PowerGraphCoefficients(df=df_power_statistics)
        if check_coefficients:
            with st.expander('АНАЛИЗ ГРАФИКОВ НАГРУЗОК'):
                st.markdown('##### Определение основных физических величин графиков нагрузки')
                check_mean_power = st.checkbox('Расчет средней месячной нагрузки')
                if check_mean_power:
                    st.markdown('###### Средняя нагрузка')
                    st.write('Средняя нагрузка – это постоянная, неизменная величина за любой рассматриваемый '
                             'промежуток времени, которая вызывает такой же расход электроэнергии, '
                             'как и изменяющаяся за это время нагрузка.')
                    st.latex(r'''P_с =  \frac {\sum_{i=1}^{n} P_{сi} }  {N}''')
                    df_mean = power_coefficients.calculation_mean_power_of_month().astype(str)
                    st.write('Результаты расчета')
                    st.write(df_mean)

                    #Запись результатов в файл
                    file_xlsx = write_to_excel(df_mean)
                    st.download_button(label='Сохранить результаты в xlsx файл',
                                      data=file_xlsx,
                                      file_name='Результаты расчета средней нагрузки.xlsx')











    except Exception as e:
        st.text(traceback.format_exc())






if __name__ == '__main__':
    app()