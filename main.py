import streamlit as st
import pandas as pd
import random
import plotly.graph_objs as go
import package_power_analytics
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

def plotly(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    return fig



#Создание шаблона Эксель файла для ввода исходных данных
def create_template_input_data(start_time, finish_time):

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
    with pd.ExcelWriter(buffer, mode='w', engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Получасовая статистика', index=False)
        df2.to_excel(writer, sheet_name='Заявл мощность', index=False)
        df3.to_excel(writer, sheet_name='Исх данные', index=False)

    writer.save()
    processed_data = buffer.getvalue()
    return processed_data


def app():
    st.markdown('### АНАЛИЗАТОР ЭЛЕКТРИЧЕСКОЙ НАГРУЗКИ')
    st.markdown('Загруженный файл в качестве исходных данных должен предусматривать ввод следующих параметров:')
    try:
        st.sidebar.header('Настройки аналитической среды')
        st.sidebar.markdown('#### Способ ввода исходных данных')
        option = st.sidebar.selectbox(label='Укажите способ ввода исходных данных',
                                  options=['Использовать шаблон Excel-файла',
                                           'Использовать ручной ввод данных'])
        if option == 'Использовать шаблон Excel-файла':
            year_now = datetime.datetime.now().year
            year_last = year_now - 1
            year_now = datetime.datetime.now().year
            year_last = year_now - 1
            st.sidebar.markdown('#### Генерация шаблона')
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.sidebar.date_input('Начальная дата', value=datetime.date(year_last, 1, 1))
            with col2:
                finish_time = st.sidebar.date_input('Конечная дата', value=datetime.date(year_last, 12, 31))

            file_xlsx = create_template_input_data(start_time, finish_time)
            st.sidebar.download_button(label='Сгенерировать шаблон для ввода данных Excel-файла',
                                       data=file_xlsx,
                                       file_name='Шаблон ввода исходных данных.xlsx')
        with st.expander('ИСХОДНЫЕ ДАННЫЕ ПРОЕКТА'):
            if option == 'Использовать шаблон Excel-файла':
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
                            st.subheader('Выбранный шаблон не соответсвует базовому. '
                                         'Скачайте и заполните шаблон для ввода данных.')
                            raise Exception

                    # Заполнение информационного поля ввода исходных данных
                    st.markdown('##### Исходные данные')
                    df_initial_data = xls.parse('Исх данные')
                    for row_index,row in df_initial_data.iterrows():
                        st.markdown(f'###### {row[0]}')
                        if row[1] is not np.nan:
                            st.text(row[1])
                        else:
                            st.markdown('Отсутсвуют данные, которые должны быть ввведены в загруженном шаблоне.')
                            raise Exception

                    # Проверка на наличие данных заявленной мощности
                    # st.markdown('##### Заявленная мощность [кВт]')
                    df_declared = xls.parse('Заявл мощность')
                    for row_index, row in df_declared.iterrows():
                        # st.markdown(f'###### {row[1]}')
                        if row[2] is not np.nan:
                            # st.text(row[2])
                            pass
                        else:
                            st.markdown('Отсутсвуют данные заявленной мощности')
                            raise Exception

                    df_power_statistics = xls.parse('Получасовая статистика')
                    if df_power_statistics.isnull().values.any() == True:
                        st.markdown('**Получасовая статистика активной и реактивной мощности не должна'
                                    'содержать пропусков!**')
                        raise Exception ('Ошибка. Не заполнены данные в исходной статистике.')

            elif option == 'Использовать ручной ввод данных':
                st.markdown('##### Заявленная мощность [кВт]')

                # col1, col2, col3, col4 = st.columns(4)
                # list_col = [col1, col2, col3, col4]
                # n = 3
                # for col in list_col:
                #     with col:
                #         st.number_input(label='', value=0)
                #         n += 1
                st.download_button(label='Сохранить изменения в Excel-файл',
                                   data='',
                                   file_name='Исходных данных.xlsx')

        with st.expander('ИСХОДНЫЕ ГРАФИКИ НАГРУЗКИ'):
            pass





    except Exception as e:
        st.text(traceback.format_exc())



if __name__ == '__main__':
    app()