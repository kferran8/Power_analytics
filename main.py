import streamlit as st
import pandas as pd
from package_power_analytics.template import create_template_input_data
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


def app():
    st.markdown('### АНАЛИЗАТОР ЭЛЕКТРИЧЕСКОЙ НАГРУЗКИ')
    try:
        st.sidebar.header('Настройки аналитической среды')
        st.sidebar.markdown('#### Редактирование исходных данных')



    except Exception as e:
        st.text(traceback.format_exc())

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

            file_xlsx_new = create_template_input_data(start_time=None, finish_time=None, df1=df_power_statistics,
                                                   df2=df_declared, df3=df_initial_data)
            st.sidebar.download_button(label='Сохранить изменения в Excel-файл',
                               data=file_xlsx_new, file_name=f'Исходные данные {df_initial_data.iloc[0,1]}.xlsx')

        with st.expander('ИСХОДНЫЕ ГРАФИКИ НАГРУЗКИ'):
            pass





    except Exception as e:
        st.text(traceback.format_exc())






if __name__ == '__main__':
    app()