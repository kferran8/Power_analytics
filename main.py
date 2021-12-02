import streamlit as st
import pandas as pd
import random
import plotly.graph_objs as go
import package_power_analytics
import os
import datetime
import traceback
from io import BytesIO
import base64



# Как развернуть приложение можно почитать здесь
# https://www.analyticsvidhya.com/blog/2021/06/deploy-your-ml-dl-streamlit-application-on-heroku/

def plotly():
    pass


# запись результата в эксель
def write_to_excel(df_initial, df_new, df_agg):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    with writer as writer:
        df_initial.to_excel(writer, sheet_name='Исходная статистика', index=False)
        df_new.to_excel(writer, sheet_name='Кластеризация', index=False)
        df_agg.to_excel(writer, sheet_name='Групповые результаты', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Функция позволяющая сделать download на выгрузку данных расчета
@st.cache
def get_table_download_link(df_initial, df_new, df_agg):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = write_to_excel(df_initial, df_new, df_agg)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Result_cluster.xlsx">' \
           f'Скачать xlsx файл результата</a>'  # decode b'abc' => abc


#Создание шаблона Эксель файла для ввода исходных данных
def create_template_input_data():
    year_now = datetime.datetime.now().year
    date = pd.date_range(start=f'{year_now - 1}-01-01', end=f'{year_now}-01-01', freq='30min')[:-1]
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

    file_name_df2 = 'Шаблон ввода исходных данных.xlsx'
    with pd.ExcelWriter(file_name_df2, mode='w') as writer:
        df1.to_excel(writer, sheet_name='Получасовая статистика', index=False)
        df2.to_excel(writer, sheet_name='Заявл мощность', index=False)
        df3.to_excel(writer, sheet_name='Исх данные', index=False)


def app():
    # st.title('Инструменты анализа электрической нагрузки')
    # st.subheader('Загрузка получасового расхода активной и реактивной энергии')
    st.markdown('### АНАЛИЗАТОР ЭЛЕКТРИЧЕСКОЙ НАГРУЗКИ')

    # st.markdown('Загруженный файл в качестве исходных данных должен предусматривать ввод следующих параметров:')
    # st.markdown('+ дату - в формате: гггг-мм-дд чч:мм, '
    #
    #             # ' * получасовую усредненную активную мощность в кВт, '
    #             # ' * получасовую усредненную реактивную мощность в кВАр')

    try:
        # uploaded_file = st.file_uploader(label='Загрузите или перетащите файл Excel ', type=['xlsx', 'xls'])
        uploaded_file = os.path.abspath('Исходная статистика\Анализируемая статистика.xlsx')
        xls = pd.ExcelFile(uploaded_file)

        sheet_names = xls.sheet_names

        ##### Настройки аналитической среды #####
        st.sidebar.header('Настройки аналитической среды')
        st.sidebar.subheader('Параметры загруженного файла')
        select_sheet_names = st.sidebar.selectbox("Выберите лист загруженного файла", (sheet_names))
        st.sidebar.text(f'Выбран лист: {select_sheet_names}')
        df_initial = pd.read_excel(uploaded_file, sheet_name=select_sheet_names)

        option = st.selectbox(label='Укажите способ ввода исходных данных',
                              options = ['Использовать шаблон Excel-файла', 'Использовать ручной ввод данных'])

        if option == 'Использовать шаблон Excel-файла':
            create_template_input_data()
            st.download_button(label='Скачать шаблон Excel-файла',)





        # with st.expander('Ввод исходных данных'):
        #     st.write(df_initial)
        #
        # with st.expander('Посмотреть результаты загруженные данных'):
        #     st.write(df_initial)









    except Exception as e:
        st.text('Необходимо загрузить файл...')
        # st.text(traceback.format_exc())

    else:
        try:
            print('')








        except Exception:
            st.subheader('Некоректно введены данные!')


if __name__ == '__main__':
    app()
