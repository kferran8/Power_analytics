import streamlit as st
import pandas as pd
import package_power_analytics
import os
import traceback

#Как развернуть приложение можно почитать здесь
#https://www.analyticsvidhya.com/blog/2021/06/deploy-your-ml-dl-streamlit-application-on-heroku/

def trying():
    pass

def app():
    # st.title('Инструменты анализа электрической нагрузки')
    # st.subheader('Загрузка получасового расхода активной и реактивной энергии')
    st.markdown('### АНАЛИЗАТОР ЭЛЕКТРИЧЕСКОЙ НАГРУЗКИ')
    # st.markdown('Загруженный файл в качестве исходных данных должен предусматривать ввод следующих параметров: \n'
    #             ' * дату - в формате: гггг-мм-дд чч:мм, '
    #             ' * получасовую усредненную активную мощность в кВт, '
    #             ' * получасовую усредненную реактивную мощность в кВАр')

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
        with st.expander('Посмотреть результаты загруженные данных'):
            st.write(df_initial)






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
