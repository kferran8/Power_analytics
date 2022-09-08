import pandas as pd
import numpy as np
import sys
from io import BytesIO



global_result = pd.DataFrame()


# Функция для описательной статистики
def describe_statistics(one_d_df):
    """Входные данные Одномерный датафрейм"""
    x = one_d_df.iloc[:,0]


    data = [['Всего', x.count()],
            ['Минимум', x.min()],
            ['Минимальная позиция', x.idxmin()],
            ['25% квантиль', round(x.quantile(.25))],
            ['Медиана', round(x.median())],
            ['75% квантиль', round(x.quantile(.75))],
            ['Среднее', round(x.mean())],
            ['Максимум', round(x.max())],
            ['Индекс максимальнрого значения',  x.idxmax()],
            ['Среднее абсолютное отклонение',  round(x.mad())],
            ['Дисперсия', round(x.var())],
            ['Среднеквадратичное отклонение', round(x.std())],
            ['Асимметрия', round(x.skew())],
            ['Эксцесс', round(x.kurt())]]

    df_result = pd.DataFrame(data, columns=['Наименование статистики', f'Значение (\n{one_d_df.columns[0]})'])
    return df_result


class PowerGraphCoefficients:

    def __init__(self, df):
        """
        :param df: Исходный DataFrame с тремя колонками: время, активная и реактивная мощность
        """
        self.df = self._timedelta(df)
        self.df_rename = self._add_period(self._rename(self.df).set_index('Период наблюдений'))
        self.count_day = round(len(self.df_rename.index) / 48)
        self.df_coef_fi_day = None
        self.df_mean_power = None
        self.df_square_power = None
        self.df_max_power = None
        self.df_coefficient_max = None
        self.df_coefficient_fill = None
        self.df_coefficient_shape = None
        self.df_coefficient_fi = None

    def _timedelta(self, df):
        df = round(df.fillna(0), 2)
        def rule(datatime):
            if (datatime.hour == 0) and (datatime.minute == 0):
                return datatime - pd.Timedelta(seconds=1)
            else:
                return datatime
        df['Дата'] = df.apply(lambda x: rule(x['Дата']), axis=1)
        return df


    def _rename(self, df):
        """Переименовывает колонки датафрейма в более удобный вид"""
        try:
            df = df.rename(columns={df.columns[0]: 'Период наблюдений', df.columns[1]: 'Активная мощность, кВт',
                                    df.columns[2]: 'Реактивная мощность, кВАр'})
        except IndexError:
            print('ВНИМАНИЕ!')
            print('Исходные данные должны быть представлены в формате xlsx виде 3 столбцов с заголовками:')
            print('Столбец 1 - время в формате ДД.ММ.ГГГГ чч:мм')
            print('Столбец 2 - активная мощность, кВт')
            print('Столбец 3 - реактивная мощность, кВар')
            print('ВВЕДИТЕ ИСХОДНЫЕ ДАННЫЕ В НУЖНОМ ФОРМАТЕ!')
            sys.exit()
        return df

    def _rename_month_name_to_rus(self, nomber_month):
        """Переименовывает названия месяцев с английского языка на русский. Входные переменные: month - номер месяца"""
        global month_name
        month_name_list = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь',
                           'октябрь', 'ноябрь', 'декабарь']
        for key, value in enumerate(month_name_list):
            if key + 1 != nomber_month:
                continue
            else:
                month_name = value
                break
        return month_name

    # Добавляем в датафрейм еще столбцы с индикаторами года, месяца, дня, часа, минуты
    def _add_period(self, df):
        df['Год'] = df.index.year
        df['Месяц'] = df.index.month
        df['День'] = df.index.day
        df['Час'] = df.index.hour
        df['Минута'] = df.index.minute
        df['Час'] = df['Час'] + df['Минута'] / 60
        return df

    def calculation_mean_power_of_month(self):
        """Считаем средюю активную и реактивную мощность по месяцам"""
        df = self._rename(self.df)
        df['Средняя полная мощность, кВА'] = \
            round(np.sqrt(df['Активная мощность, кВт'] ** 2 + df['Реактивная мощность, кВАр'] ** 2), 2)
        df = df.rename(columns={df.columns[1]: 'Средняя активная мощность, кВт',
                                df.columns[2]: 'Средняя реактивная мощность, кВар'})
        df = df.set_index('Период наблюдений')

        df = df.resample('M').mean().reset_index(inplace=False)
        df = round(df, 2)
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        df_mean_power = df
        self.df_mean_power = df_mean_power.dropna()
        return df_mean_power.dropna()

    def calculation_square_power_of_month(self):
        df = self._rename(self.df)
        df = df.rename(columns={df.columns[1]: 'Среднеквадратичная активная мощность, кВт',
                                df.columns[2]: 'Среднеквадратичная реактивная мощность, кВар'})
        df = df.set_index('Период наблюдений')
        df_count = df.resample('M').count()  # Количество наблюдений в месяце
        df = df.iloc[:, [0, 1]].apply(lambda x: x ** 2)  # Возвел в квадрат каждое значение
        df_sum_month = df.resample('M').sum()  # Среднее наблюдений в месяце
        df_square_power = round(np.sqrt(df_sum_month / df_count), 1)
        df_square_power['Среднеквадратичная полная мощность, кВА'] = \
            round(np.sqrt(df_square_power.iloc[:, 0] ** 2 + df_square_power.iloc[:, 1] ** 2))
        df_square_power = df_square_power.reset_index(inplace=False)
        df_square_power['Период наблюдений'] = df_square_power['Период наблюдений'].dt.to_period('M')
        self.df_square_power = df_square_power.dropna()
        return df_square_power.dropna()

    def calculation_max_power_of_month(self):
        df = self._rename(self.df)
        df = df.rename(columns={df.columns[1]: 'Максимальная активная мощность, кВт',
                                df.columns[2]: 'Максимальная реактивная мощность, кВар', })
        df = df.set_index('Период наблюдений')
        df_max = round(df.resample('M').max(), 2)
        df_max['Максимальная полная мощность, кВА'] = \
            round(np.sqrt(df_max.iloc[:, 0] ** 2 + df_max.iloc[:, 1] ** 2))
        df_max = df_max.reset_index(inplace=False)
        df_max['Период наблюдений'] = df_max['Период наблюдений'].dt.to_period('M')
        self.df_max_power = df_max.dropna()
        return df_max.dropna()

    def coefficient_max(self):
        """Рассчитывает коэффициент максимума нагрузки по месяцам"""
        df = self._rename(self.df).set_index('Период наблюдений').iloc[:, [0]]
        df_max_day = df.resample('D').max()
        df_mean_max_power_in_month = df_max_day.resample('M').mean()
        df_mean_power_in_month = df.resample('M').mean()
        coefficient_max = round(df_mean_max_power_in_month / df_mean_power_in_month, 2)
        coefficient_max = coefficient_max.rename(columns={df.columns[0]: 'Коэффициент максимума'}). \
            reset_index(inplace=False)
        coefficient_max['Период наблюдений'] = coefficient_max['Период наблюдений'].dt.to_period('M')
        self.df_coefficient_max = coefficient_max
        return coefficient_max

    def coefficient_fill(self):
        """Коэффициент заполнения графика нагрузок"""
        if self.df_coefficient_max is not None:
            pass
        else:
            self.coefficient_max()

        coefficient_max = self.df_coefficient_max.set_index('Период наблюдений')
        coefficient_fill = pd.DataFrame()
        coefficient_fill['Коэффициент заполнения'] = round(coefficient_max['Коэффициент максимума'].
                                                           apply(lambda x: 1 / x), 2)
        self.df_coefficient_fill = coefficient_fill.reset_index(inplace=False).dropna()
        coefficient_fill = self.df_coefficient_fill
        return coefficient_fill.dropna()

    def coefficient_shape(self):
        """Коэффициент формы графика нагрузок"""
        if self.df_square_power is not None:
            pass
        else:
            self.calculation_square_power_of_month()
        if self.df_mean_power is not None:
            pass
        else:
            self.calculation_mean_power_of_month()
        df_square = self.df_square_power.set_index('Период наблюдений')
        df_mean = self.df_mean_power.set_index('Период наблюдений')
        coefficient_shape = pd.DataFrame()
        coefficient_shape['Коэффициент формы'] = round(df_square.iloc[:, 0] / df_mean.iloc[:, 0], 3)
        coefficient_shape = coefficient_shape.reset_index(inplace=False).dropna()
        self.df_coefficient_shape = coefficient_shape
        return coefficient_shape

    def coefficient_fi(self):
        df = self._rename(self.df)
        df = df.rename(columns={df.columns[1]: 'Активная мощность, кВт',
                                df.columns[2]: 'Реактивная мощность, кВар',
                                }
                       )
        df = df.set_index('Период наблюдений')
        df['Полная мощность, кВА'] = \
            round(np.sqrt(df.iloc[:, 0] ** 2 + df.iloc[:, 1] ** 2))
        df['Коэффициент активной мощности'] = df.iloc[:, 0] / df.iloc[:, 2]
        self.df_coef_fi_day = round(df.drop(['Активная мощность, кВт', 'Реактивная мощность, кВар',
                                             'Полная мощность, кВА'], axis='columns'), 3)
        self.df_coef_fi_day = self.df_coef_fi_day.rename(columns={'Коэффициент активной мощности':
                                                                      'Суточный коэффициент активной мощности'
                                                                  })
        df = df.resample('M').mean()
        df = df.reset_index(inplace=False)
        coefficient_fi = round(df.drop(['Активная мощность, кВт', 'Реактивная мощность, кВар',
                                        'Полная мощность, кВА'], axis='columns'), 3)
        coefficient_fi['Период наблюдений'] = coefficient_fi['Период наблюдений'].dt.to_period('M')
        self.df_coefficient_fi = coefficient_fi.dropna()
        return coefficient_fi.dropna()

    def write_to_exel_buffer(self):

        if self.df_mean_power is not None:
            pass
        else:
            self.calculation_mean_power_of_month()
        if self.df_square_power is not None:
            pass
        else:
            self.calculation_square_power_of_month()
        if self.df_max_power is not None:
            pass
        else:
            self.calculation_max_power_of_month()
        if self.df_coefficient_max is not None:
            pass
        else:
            self.coefficient_max()
        if self.df_coefficient_fill is not None:
            pass
        else:
            self.coefficient_fill()
        if self.df_coefficient_shape is not None:
            pass
        else:
            self.coefficient_shape()
        if self.df_coefficient_fi is not None:
            pass
        else:
            self.coefficient_fi()

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            self.df_mean_power.to_excel(writer, sheet_name='Средняя мощность', index=False)
            self.df_square_power.to_excel(writer, sheet_name='Среднеквадратичная мощность', index=False)
            self.df_max_power.to_excel(writer, sheet_name='Максимальная мощность', index=False)
            self.df_coefficient_max.to_excel(writer, sheet_name='Коэф. максимума', index=False)
            self.df_coefficient_fill.to_excel(writer, sheet_name='Коэф. заполнения', index=False)
            self.df_coefficient_shape.to_excel(writer, sheet_name='Коэф. формы', index=False)
            self.df_coefficient_fi.to_excel(writer, sheet_name='Косинус фи', index=False)
        processed_data = buffer.getvalue()
        return processed_data


class DTariff(PowerGraphCoefficients):
    """
    Расчет стоимости оплаты по Д-тарифу с фактической мощностью
    :param df: Исходный DataFrame с тремя колонками: время, активная и реактивная мощность
    :param ab: основная ставка;
    :param bb: дополнительная ставка
    :param kt: текущий курс доллара
    :param kb: базовый курс доллара
    """

    def __init__(self, df, ab, bb, kt, kb, time_start_morning =  8.5, time_finish_morning = 11,
                 time_start_evening = 18.5, time_finish_evening = 21):
        super().__init__(df)
        self.ab = ab
        self.bb = bb
        self.kt = kt
        self.kb = kb
        self.time_start_morning = time_start_morning
        self.time_finish_morning = time_finish_morning
        self.time_start_evening = time_start_evening
        self.time_finish_evening = time_finish_evening
        self.df_pay_energy_month = None
        self.df_pay_energy_day = None
        self.sum_pay_power = None
        self.sum_pay_energy = None
        self.sum_pay_power_and_energy = None
        self.sum_energy = None
        self.mean_tariff = None  # Средний тариф оплаты за электроэнергию
        self.tariff_power = round(self.ab * (0.31 + 0.69 * self.kt / self.kb), 4)  # Тариф без НДС
        self.tariff_energy = round(self.bb * (0.31 + 0.69 * self.kt / self.kb), 4)  # Тариф без НДС

    def power_analyzer_day(self):
        """Результат метода: Датафрейм:
        Индекс: период по месяцам и суткам.
        Столбцы: Максимум наблюдавшейся активной мощности, кВт',
       'Максимум активной мощности в часы утреннего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы вечернего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы максимума энергосистемы, кВт',
       'Индикатор: Наибольший максимум активной мощности выходит за границы максимумов энергосистемы?.
                   True - да, False - нет """
        df = self.df_rename.reset_index()
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        df_max_power_in_day = df.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'max'})
        df_max_power_in_day = df_max_power_in_day.rename(columns={'Активная мощность, кВт':
                                                                      'Максимум наблюдавшейся активной мощности, кВт'})
        # Максимальная мощность в только в часы максимума (по этой мощности оплата)
        x = df['Час']
        # Максимально-вероятностный период в часы утреннего максимума
        df_full_in_morning_day = df[((x >= self.time_start_morning) & (x <= self.time_finish_morning))]
        df_max_power_in_morning_day = df_full_in_morning_day.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'max'})
        df_max_power_in_morning_day = df_max_power_in_morning_day. \
            rename(columns={'Активная мощность, кВт': 'Максимум активной мощности в часы утреннего максимума '
                                                      'энергосистемы , кВт'})
        # Максимально-вероятностный период в часы вечернего максимума
        df_full_in_evening_day = df[((x >= self.time_start_evening) & (x <= self.time_finish_evening))]
        df_max_power_in_evening_day = df_full_in_evening_day.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'max'})
        df_max_power_in_evening_day = df_max_power_in_evening_day. \
            rename(columns={'Активная мощность, кВт': 'Максимум активной мощности в часы вечернего максимума '
                                                      'энергосистемы , кВт'})
        df_full_in_hour_energy_day = df[((x >= self.time_start_morning) & (x <= self.time_finish_morning)) |
                                        (x >= self.time_start_evening) & (x <= self.time_finish_evening)]

        df_full_in_hour_energy_day = df_full_in_hour_energy_day.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'max'})
        df_full_in_hour_energy_day = df_full_in_hour_energy_day. \
            rename(columns={'Активная мощность, кВт': 'Максимум активной мощности в часы максимума энергосистемы, кВт'})
        # Создаем единый фрейм данных
        df_max_even_mor_pow_day = pd.concat([df_max_power_in_day, df_max_power_in_morning_day,
                                             df_max_power_in_evening_day, df_full_in_hour_energy_day],
                                            axis=1)
        df_max_even_mor_pow_day['Индикатор'] = \
            (df_max_even_mor_pow_day.iloc[:, 0] > df_max_even_mor_pow_day.iloc[:, 1]) & \
            (df_max_even_mor_pow_day.iloc[:, 0] > df_max_even_mor_pow_day.iloc[:, 2])

        return df_max_even_mor_pow_day

    def power_analyzer_month(self):
        """Результат метода: Датафрейм:
        Индекс: период по месяцам.
        Столбцы: Максимум наблюдавшейся активной мощности, кВт',
       'Максимум активной мощности в часы утреннего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы вечернего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы максимума энергосистемы, кВт',
       'Индикатор: Наибольший максимум активной мощности выходит за границы максимумов энергосистемы?.
                   True - да, False - нет ' """
        df = self.power_analyzer_day()
        df = df.reset_index().drop(['День'], axis=1)
        df = df.groupby('Период наблюдений').agg({'Максимум наблюдавшейся активной мощности, кВт': 'max',
                                                  'Максимум активной мощности в часы утреннего максимума энергосистемы , кВт': 'max',
                                                  'Максимум активной мощности в часы вечернего максимума энергосистемы , кВт': 'max',
                                                  'Максимум активной мощности в часы максимума энергосистемы, кВт': 'max'
                                                  })
        df['Индикатор'] = (df.iloc[:, 0] > df.iloc[:, 1]) & (df.iloc[:, 0] > df.iloc[:, 2])
        df_max_even_mor_pow_month = df
        return df_max_even_mor_pow_month

    def energy_analyzer_day(self):
        """Результат метода: Датафрейм:
        Индекс: период наблюдений, день.
        Столбцы: Количество значений 30-и минутной мощности попавших в диапазон',
       'Количество дней исследования',
       'Расход электроэнергии, кВтч """
        df = self.df_rename.reset_index()
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        df['Количество дней в месяце'] = df['Период наблюдений'].apply(lambda x: x.day)
        # Расход элекроэнергии
        df['Расход электроэнергии, кВтч'] = df['Активная мощность, кВт'].apply(lambda x: round(0.5 * x, 1))
        df_energy_day = df.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'count', 'День': 'count', 'Количество дней в месяце': 'mean',
                       'Расход электроэнергии, кВтч': 'sum'})

        df_energy_day['День'] = df_energy_day['День'].apply(lambda x: round(x / (48),1))
        df_energy_day = df_energy_day.rename(columns={'Активная мощность, кВт':
                                                          'Количество значений 30-и минутной '
                                                          'мощности попавших в диапазон',
                                                      'День': 'Количество исследованных дней'})
        return df_energy_day

    def energy_analyzer_month(self):
        """Результат метода: Датафрейм:
        Индекс: период наблюдений.
        Столбцы: 'Количество значений 30-и минутной мощности попавших в диапазон',
       'Количество исследованных дней', 'Количество дней в месяце',
       'Расход электроэнергии, кВтч' """
        df = self.energy_analyzer_day()
        df = df.reset_index().drop(['День'], axis=1)
        df_energy_month = df.groupby('Период наблюдений').agg({'Количество значений 30-и минутной мощности попавших в '
                                                               'диапазон': 'sum',
                                                               'Количество исследованных дней': 'sum',
                                                               'Количество дней в месяце': 'mean',
                                                               'Расход электроэнергии, кВтч': 'sum'
                                                               })
        self.sum_energy = df['Расход электроэнергии, кВтч'].sum()
        return df_energy_month

    def calculation(self):
        """Результат метода вывод опталы по суткам и месяцам:
         self.sum_pay_power - датафрейм по месяцам: pay_energy_month:
         Период наблюдений;	Количество исследованных дней;	Максимум активной мощности в часы максимума энергосистемы, кВт
        Тариф на электрическую мощность, руб/кВт;	Оплата за электрическую мощность, руб;
        Расход электроэнергии, кВтч;	Тариф на электрическую энергию, руб/кВтч; Оплата за электрическую энергию, руб;
        Суммарная оплата за электроэнергию, руб, Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч
         """

        # Расчет оплаты по суткам
        pay_energy_day = pd.concat([self.energy_analyzer_day(), self.power_analyzer_day()], axis=1)
        pay_energy_day['Тариф на электрическую мощность, руб/кВт'] = self.tariff_power
        pay_energy_day['Тариф на электрическую энергию, руб/кВтч'] = self.tariff_energy
        max_power_day = pay_energy_day.groupby(['Период наблюдений', 'День']).aggregate(
            {'Максимум активной мощности в часы максимума энергосистемы, кВт': max})
        max_power_month = pay_energy_day.groupby(['Период наблюдений']).aggregate(
            {'Максимум активной мощности в часы максимума энергосистемы, кВт': max})
        max_power_month = max_power_month.rename(columns={
            'Максимум активной мощности в часы максимума энергосистемы, кВт': 'Максимум активной мощности, кВт'})
        merge_power = max_power_day.join(max_power_month)
        pay_energy_day['Максимум активной мощности в месяце в часы максимума энергосистемы, кВт'] = merge_power[
            'Максимум активной мощности, кВт']
        pay_energy_day['Оплата за электрическую мощность, руб'] = round(
            pay_energy_day['Тариф на электрическую мощность, руб/кВт'] * \
            pay_energy_day['Максимум активной мощности в месяце в часы максимума энергосистемы, кВт'] / pay_energy_day[
                'Количество дней в месяце'],2)

        pay_energy_day['Оплата за электрическую энергию, руб'] = round(pay_energy_day[
                                                                           'Тариф на электрическую энергию, руб/кВтч'] * \
                                                                       pay_energy_day[
                                                                           'Расход электроэнергии, кВтч'],2)
        pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'] = round(pay_energy_day['Оплата за электрическую ' \
                                                                                              'мощность, руб'] + \
                                                                               pay_energy_day[
                                                                                   'Оплата за электрическую энергию, ' \
                                                                                   'руб'],2)
        pay_energy_day['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] \
            = round(pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'] / \
              pay_energy_day['Расход электроэнергии, кВтч'],4)

        self.mean_tariff = round(pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'].sum() / \
                                 pay_energy_day['Расход электроэнергии, кВтч'].sum(), 4)

        self.df_pay_energy_day = pay_energy_day.iloc[:, [1, 2, 11, 9, 12, 3, 10, 13, 14, 15]]

        # Расчет оплаты по месяцам
        pay_energy_month = self.df_pay_energy_day.reset_index().set_index('Период наблюдений'). \
            groupby('Период наблюдений'). \
            aggregate({'Количество исследованных дней': 'sum',
                       'Количество дней в месяце': 'mean',
                       'Максимум активной мощности в месяце в часы максимума энергосистемы, кВт':
                           lambda x: np.round(np.mean(x), 2),
                       'Тариф на электрическую мощность, руб/кВт': lambda x: np.round(np.mean(x), 5),
                       'Оплата за электрическую мощность, руб': 'sum',
                       'Расход электроэнергии, кВтч': 'sum',
                       'Тариф на электрическую энергию, руб/кВтч': lambda x: np.round(np.mean(x), 5),
                       'Оплата за электрическую энергию, руб': 'sum',
                       'Суммарная оплата за электроэнергию и мощность, руб': lambda x: np.round(np.sum(x), 2),
                       })
        pay_energy_month['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] = \
           round( pay_energy_month['Суммарная оплата за электроэнергию и мощность, руб'] / \
            pay_energy_month['Расход электроэнергии, кВтч'],4)

        self.df_pay_energy_month = pay_energy_month
        self.sum_pay_power = float(round(self.df_pay_energy_month.iloc[:, [4]].sum()))
        self.sum_pay_energy = float(round(self.df_pay_energy_month.iloc[:, [7]].sum()))
        self.sum_pay_power_and_energy = float(round(self.df_pay_energy_month.iloc[:, [8]].sum()))

    def write_to_exel_buffer(self):

        if self.df_pay_energy_day is not None:
            pass
        else:
            self.calculation()
        if self.df_pay_energy_month is not None:
            pass
        else:
            self.calculation()

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            self.energy_analyzer_day().reset_index().to_excel(writer, sheet_name='Расход ЭЭ (день).xlsx', index=False)
            self.energy_analyzer_month().reset_index().to_excel(writer, sheet_name='Расход ЭЭ (месяц).xlsx', index=False)
            self.power_analyzer_day().reset_index().to_excel(writer, sheet_name='Макс акт мощности (день).xlsx', index=False)
            self.power_analyzer_month().reset_index().to_excel(writer, sheet_name='Макс акт мощности (месяц).xlsx', index=False)
            self.df_pay_energy_day.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по суткам.xlsx', index=False)
            self.df_pay_energy_month.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по месяцам.xlsx', index=False)
        processed_data = buffer.getvalue()
        return processed_data


class PowerLimits(PowerGraphCoefficients):

    def __init__(self, df, time_start_morning =  8.5, time_finish_morning = 11,
                 time_start_evening = 18.5, time_finish_evening = 21):
        super().__init__(df)
        self.time_start_morning = time_start_morning
        self.time_finish_morning = time_finish_morning
        self.time_start_evening = time_start_evening
        self.time_finish_evening = time_finish_evening
        self.df_full_limit = None
        self.df_only_max_period = None
        self.df_max_month_value = None

    def power_limits(self):
        """Результат метода датафрейм с колонками:
        Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток',
       'Средняя активная мощность, кВт', 'Минимальная активная мощность, кВт',
       'Максимальная активная мощность, кВт',
       'Среднеквадратическое отклонение активной мощности, кВт',
       'Количество значений',
       'Лимит активной мощности с 5% вероятностью, кВт'
         """
        df = self.df_rename

        def hour_and_minute(df):
            hour = str(df.hour)
            minute = str(df.minute)
            h_m = hour + ':' + minute
            return h_m

        df = df.reset_index()
        df['Часы суток'] = df['Период наблюдений'].apply(lambda x: hour_and_minute(x))
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        df_mean = df.groupby(['Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток', ]).aggregate(
            {'Активная мощность, кВт': 'mean'}). \
            rename(columns={'Активная мощность, кВт': 'Средняя активная мощность, кВт'})

        df1 = df_mean
        df_min = df.groupby(['Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток']).aggregate(
            {'Активная мощность, кВт': 'min'})
        df_max = df.groupby(['Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток']).aggregate(
            {'Активная мощность, кВт': 'max'})
        df_std = df.groupby(['Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток']).aggregate(
            {'Активная мощность, кВт': 'std'})
        df_count = df.groupby(['Период наблюдений', 'Год', 'Месяц', 'Час', 'Часы суток']). \
            aggregate({'Активная мощность, кВт': 'count'}). \
            rename(columns={'Активная мощность, кВт': 'Количество значений мощности в исследуемом интервале'})
        df1['Минимальная активная мощность, кВт'] = df_min
        df1['Максимальная активная мощность, кВт'] = df_max
        df1['Среднеквадратическое отклонение активной мощности, кВт'] = df_std
        df1['Количество значений'] = df_count

        def interval_high_level(row):
            # 1.65 - для нормального распределения статистический коэффициент 5-% вероятности
            x = row['Средняя активная мощность, кВт'] + \
                1.65 * row['Среднеквадратическое отклонение активной мощности, кВт']
            return x

        interval_high_list = []
        for index, row in df1.reset_index('Месяц').iterrows():
            x = interval_high_level(row=row)
            interval_high_list.append(x)
        df2 = pd.DataFrame(interval_high_list, columns=['Лимит активной мощности с 5% вероятностью, кВт'],
                           index=df1.index)

        df_full_limit = df1.join(df2)
        self.df_full_limit = round(df_full_limit, 2)
        self.df_full_limit = self.df_full_limit.reset_index()

        # Максимально-вероятностные значения с русскими месяцами
        self.df_full_limit['Месяц'] = self.df_full_limit['Месяц']. \
            apply(lambda x: self._rename_month_name_to_rus(x))
        x = self.df_full_limit['Час']
        #
        # Максимально-вероятностный период в часы максимума
        self.df_only_max_period = self.df_full_limit[((x >= self.time_start_morning) & (x <= self.time_finish_morning))
                                                     | (x >= self.time_start_evening) & (x <= self.time_finish_evening)]

        # Только максимально-вероятностное значения в часы максимума
        self.df_max_month_value = self.df_only_max_period.groupby('Период наблюдений'). \
            aggregate({'Лимит активной мощности с 5% вероятностью, кВт': 'max',
                       'Максимальная активная мощность, кВт': 'max'}).reset_index()
        #Для случая анализа одного дня, когда нельзя посчитать среднее квадратическое (результат Nan)
        # отклонение вместо лимита берем максимум мощности
        self.df_max_month_value.iloc[:, [1]] = self.df_max_month_value. \
            apply(lambda x: (x.iloc[2] if pd.isnull(x.iloc[1]) == True else x.iloc[1]), axis=1)

        return self.df_full_limit

    def write_to_exel_buffer(self):

        if self.df_only_max_period is not None:
            pass
        else:
            self.power_limits()


        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            self.df_only_max_period.to_excel(writer, sheet_name='Часовая оценка лимитов', index=False)
            self.df_max_month_value.to_excel(writer, sheet_name='Ожидаемые лимиты', index=False)
        processed_data = buffer.getvalue()
        return processed_data


class DTariffDeclared(DTariff, PowerLimits):
    """Расчет стоимости оплаты по Д-тарифу с заявленной мощностью"""

    def __init__(self, df, ab, bb, kt, kb, declared):
        super().__init__(df, ab, bb, kt, kb)

        # self.pay_energy_month = None
        self.declared = declared.set_index('Номер месяца').iloc[:, [1]].fillna(0)  # исходные данные эксель

    def calculation(self):
        # ______ вывод лимитов мощности _________#
        self.power_limits()  # Запустили расчет лимитов
        lim = self.df_max_month_value.set_index('Период наблюдений')  # Вывели свойство лимитов
        lim = lim.reset_index()

        lim['Номер месяца'] = lim['Период наблюдений'].apply(lambda x: x.month)
        lim = lim.set_index('Номер месяца').iloc[:, [0, 1]]
        limits = lim.join(self.declared)


        # limits = pd.concat([lim, self.declared], axis=1)

        limits = limits.dropna()


        # функция для вывода заявленной мощности, если она задана и вывода максимально-вероятностной, если незадана
        def rule(x, y):
            if y != 0:
                return y
            else:
                return x

        limits['Заявленная мощность, кВт'] = limits.apply(lambda x:
                                                          rule(x['Лимит активной мощности с 5% вероятностью, кВт'],
                                                               x['Заявленная мощность, кВт']), axis=1)

        lim_for_day = limits.set_index('Период наблюдений').iloc[:, [1]]
        limits = limits.iloc[:, [0, 2]].reset_index()
        limits = limits.set_index('Период наблюдений').iloc[:, [1]]

        # ___________________РАСЧЕТ ОПЛАТЫ ПО МЕСЯЦАМ ЗАЯВЛЕННОЙ МОЩНОСТИ_________________#
        df_energy_day = self.energy_analyzer_day()
        df_power_analyzer_day = self.power_analyzer_day()

        # ______________Расчет оплаты по суткам__________________
        pay_energy_day = pd.concat([df_energy_day, df_power_analyzer_day], axis=1)
        pay_energy_day['Тариф на электрическую мощность, руб/кВт'] = self.tariff_power
        pay_energy_day['Тариф на электрическую энергию, руб/кВтч'] = self.tariff_energy
        max_power_day = pay_energy_day.groupby(['Период наблюдений', 'День']).aggregate(
            {'Максимум активной мощности в часы максимума энергосистемы, кВт': max})

        merge_power = max_power_day.join(lim_for_day)
        pay_energy_day['Заявленная мощность, кВт'] = merge_power[
            'Заявленная мощность, кВт']
        pay_energy_day['Оплата за электрическую мощность, руб'] = round(
            pay_energy_day['Тариф на электрическую мощность, руб/кВт'] * \
            pay_energy_day['Заявленная мощность, кВт'] / pay_energy_day[
                'Количество дней в месяце'])
        pay_energy_day['Оплата за электрическую энергию, руб'] = round(pay_energy_day[
                                                                           'Тариф на электрическую энергию, руб/кВтч'] * \
                                                                       pay_energy_day[
                                                                           'Расход электроэнергии, кВтч'])
        pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'] = pay_energy_day['Оплата за электрическую ' \
                                                                                              'мощность, руб'] + \
                                                                               pay_energy_day[
                                                                                   'Оплата за электрическую энергию, ' \
                                                                                   'руб']
        pay_energy_day['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] \
            = round(pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'] / \
              pay_energy_day['Расход электроэнергии, кВтч'],4)

        self.mean_tariff = round(pay_energy_day['Суммарная оплата за электроэнергию и мощность, руб'].sum() / \
                                 pay_energy_day['Расход электроэнергии, кВтч'].sum(), 4)

        self.df_pay_energy_day = pay_energy_day.iloc[:, [1, 2, 11, 9, 12, 3, 10, 13, 14]]

        # Расчет оплаты по месяцам
        pay_energy_month = self.df_pay_energy_day.reset_index().set_index('Период наблюдений'). \
            groupby('Период наблюдений'). \
            aggregate({'Количество исследованных дней': 'sum',
                       'Количество дней в месяце': 'mean',
                       'Заявленная мощность, кВт':
                           lambda x: np.round(np.mean(x), 2),
                       'Тариф на электрическую мощность, руб/кВт': lambda x: np.round(np.mean(x), 4),
                       'Оплата за электрическую мощность, руб': 'sum',
                       'Расход электроэнергии, кВтч': 'sum',
                       'Тариф на электрическую энергию, руб/кВтч': lambda x: np.round(np.mean(x), 4),
                       'Оплата за электрическую энергию, руб': 'sum',
                       'Суммарная оплата за электроэнергию и мощность, руб': 'sum',
                       })
        pay_energy_month['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] = \
            round(pay_energy_month['Суммарная оплата за электроэнергию и мощность, руб'] / \
            pay_energy_month['Расход электроэнергии, кВтч'],4)
        self.df_pay_energy_month = pay_energy_month
        self.sum_pay_power = float(round(self.df_pay_energy_month.iloc[:, [4]].sum()))
        self.sum_pay_energy = float(round(self.df_pay_energy_month.iloc[:, [7]].sum()))
        self.sum_pay_power_and_energy = float(round(self.df_pay_energy_month.iloc[:, [8]].sum()))


    def write_to_exel_buffer(self):
        if self.df_pay_energy_month is not None:
            pass
        else:
            self.calculation()
        if self.df_pay_energy_day is not None:
            pass
        else:
            self.calculation()

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            self.energy_analyzer_day().reset_index().to_excel(writer, sheet_name='Расход ЭЭ (день).xlsx', index=False)
            self.energy_analyzer_month().reset_index().to_excel(writer, sheet_name='Расход ЭЭ (месяц).xlsx', index=False)
            self.power_analyzer_day().reset_index().to_excel(writer, sheet_name='Макс акт мощности (день).xlsx', index=False)
            self.power_analyzer_month().reset_index().to_excel(writer, sheet_name='Макс акт мощности (месяц).xlsx', index=False)
            self.df_pay_energy_month.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по месяцам.xlsx', index=False)
            self.df_pay_energy_day.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по суткам.xlsx', index=False)
        processed_data = buffer.getvalue()
        return processed_data


class DifferTariff(DTariff, PowerLimits):
    """Расчет стоимости оплаты по Д-тарифу с заявленной мощностью"""

    def __init__(self, df, ab, bb, kt, kb):
        super().__init__(df, ab, bb, kt, kb)
        self.tp = 3  # время пиковой зоны
        self.tpp = 14  # время полупиковой зоны
        self.tn = 7  # время ночной зоны
        self.sum_energy_night = None
        self.sum_energy_peak = None
        self.sum_energy_half_peak = None
        self.sum_energy = None
        # Параметры оплаты
        self.df_pay_energy_day = None
        self.df_pay_energy_month = None
        self.sum_pay_power = None
        self.sum_pay_energy = None
        self.sum_pay_energy_night = None
        self.sum_pay_energy_peak = None
        self.sum_pay_energy_half_peak = None
        self.sum_pay_power_and_energy = None

    def calculation_tariff_coefficients(self):
        """Результат метода: Датафрейм:
             Индекс: период по месяцам.
             Столбцы: Календарное количество дней в расчетном периоде,
            'Понижающий  коэффициент к основной ставке',
            'Тарифный ночной коэффициент,
             Тарифный полупиковый коэффициент,
             Тарифный пиковый коэффициент"""
        df = self.df_rename.reset_index()
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        # Календарное количество дней в расчетном периоде
        df['Количество дней в месяце'] = df['Период наблюдений'].apply(lambda x: x.day)
        df = df.iloc[:, [0, 8]].groupby(['Период наблюдений']).aggregate({'Количество дней в месяце': 'mean'})
        df['Понижающий коэффициент к основной ставке'] = 0.5
        df = df.reset_index()
        df['Тарифный ночной коэффициент'] = df.apply(
            lambda x: round(1 - (self.tariff_power * (1 - 0.5) * (4 * self.tp - self.tn)) /
                            (self.tariff_energy * x['Количество дней в месяце'] *
                             (self.tn ** 2 - self.tp ** 2)), 4), axis=1)
        df['Тарифный полупиковый коэффициент'] = 1.00
        df['Тарифный пиковый коэффициент'] = df.apply(
            lambda x: round(1 + (self.tariff_power * (1 - 0.5) * (4 * self.tn - self.tp)) /
                            (self.tariff_energy * x['Количество дней в месяце'] *
                             (self.tn ** 2 - self.tp ** 2)), 4), axis=1)
        df = df.set_index('Период наблюдений')
        return df

    def dd_energy_analyzer_day(self):
        """Результат метода: Датафрейм:
               Индекс: два индекса период наблюдений (2020-01) и номер суток (1,2..).
               Столбцы: Расход ЭЭ в ночной зоне по суткам,
              ' Расход ЭЭ в пиковой зоне по суткам',
              ' Расход ЭЭ в полупиковой зоне по суткам,
               Суммарный расход ЭЭ по суткам"""
        df = self.df_rename.reset_index()
        df['Период наблюдений'] = df['Период наблюдений'].dt.to_period('M')
        df['Количество дней в месяце'] = df['Период наблюдений'].apply(lambda x: x.day)
        # Расход элекроэнергии
        df['Расход электроэнергии, кВтч'] = df['Активная мощность, кВт'].apply(lambda x: round(0.5 * x, 2))
        df_energy_day = df.groupby(['Период наблюдений', 'День']). \
            aggregate({'Активная мощность, кВт': 'count', 'День': 'count', 'Количество дней в месяце': 'mean',
                       'Расход электроэнергии, кВтч': 'sum'})
        df_energy_day['День'] = df_energy_day['День'].apply(lambda x: round(x / (48), 2))
        df_energy_day = df_energy_day.rename(columns={'Активная мощность, кВт':
                                                          'Количество значений 30-и минутной '
                                                          'мощности попавших в диапазон',
                                                      'День': 'Количество исследованных дней'})
        self.sum_energy = df_energy_day['Расход электроэнергии, кВтч'].sum()

        # _________
        x = df['Час']
        df_full_in_night = df[((x <= 6) | (x >= 23.5))]  # Датафрейм в ночной зоне
        df_full_in_peak = df[((x >= 8.5) & (x <= 11.0))]  # Датафрейм в пиковой зоне
        df_full_in_half_peak = df[
            (((x >= 6.5) & (x <= 8.0)) | ((x >= 11.5) & (x <= 23.0)))]  # Датафрейм в полупиковой зоне

        # Расход ЭЭ в ночной зоне
        df_energy_night = df_full_in_night.groupby(['Период наблюдений', 'День']). \
            aggregate({'Расход электроэнергии, кВтч': 'sum'}). \
            rename(columns={'Расход электроэнергии, кВтч': 'Расход электроэнергии в ночной зоне, кВтч'})
        self.sum_energy_night = round(df_energy_night['Расход электроэнергии в ночной зоне, кВтч'].sum())

        # Расход ЭЭ в пиковой зоне
        df_energy_peak = df_full_in_peak.groupby(['Период наблюдений', 'День']). \
            aggregate({'Расход электроэнергии, кВтч': 'sum'}). \
            rename(columns={'Расход электроэнергии, кВтч': 'Расход электроэнергии в пиковой зоне, кВтч'})
        self.sum_energy_peak = round(df_energy_peak['Расход электроэнергии в пиковой зоне, кВтч'].sum())

        # Расход ЭЭ в полупиковой зоне
        df_energy_half_peak = df_full_in_half_peak.groupby(['Период наблюдений', 'День']). \
            aggregate({'Расход электроэнергии, кВтч': 'sum'}). \
            rename(columns={'Расход электроэнергии, кВтч': 'Расход электроэнергии в полупиковой зоне, кВтч'})
        self.sum_energy_half_peak = round(df_energy_half_peak['Расход электроэнергии в полупиковой зоне, кВтч'].sum())

        # Соеденяем все данные в один фрейм
        df_energy_day = pd.concat([df_energy_night, df_energy_peak, df_energy_half_peak], axis=1)
        df_energy_day['Суммарный расход электроэнергии, кВтч'] = \
            df_energy_day['Расход электроэнергии в ночной зоне, кВтч'] + \
            df_energy_day['Расход электроэнергии в пиковой зоне, кВтч'] + \
            df_energy_day['Расход электроэнергии в полупиковой зоне, кВтч']
        self.sum_energy = round(df_energy_day['Суммарный расход электроэнергии, кВтч'].sum(),0)
        return df_energy_day

    def dd_energy_analyzer_month(self):
        """Результат метода: Датафрейм:
              Индекс: период период наблблений (2020-01),.
              Столбцы: Расход ЭЭ в ночной зоне,
             ' Расход ЭЭ в пиковой зоне',
             ' Расход ЭЭ в полупиковой зоне,
              Суммарный расход ЭЭ"""
        df_energy = self.dd_energy_analyzer_day().reset_index()
        df_energy_month = df_energy.groupby('Период наблюдений'). \
            aggregate({'Расход электроэнергии в ночной зоне, кВтч': lambda x: np.round(np.sum(x), 0),
                       'Расход электроэнергии в пиковой зоне, кВтч': lambda x: np.round(np.sum(x), 0),
                       'Расход электроэнергии в полупиковой зоне, кВтч': lambda x: np.round(np.sum(x), 0),
                       'Суммарный расход электроэнергии, кВтч':  lambda x: np.round(np.sum(x), 0)})

        return df_energy_month

    def dd_power_analyzer_day(self):
        """Результат метода: Датафрейм:
            Индекс: Период наблюдений, 'День'
            Столбцы: 'Максимум наблюдавшейся активной мощности, кВт',
       'Максимум активной мощности в часы утреннего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы вечернего максимума энергосистемы , кВт',
       'Максимум активной мощности в часы максимума энергосистемы, кВт',
       'Мощность превышения вечернего максимума утренним, кВт'
       'Индикатор' (Наибольший утренний вечерний максимум больше утреннего?.
                          True - да, False - нет )"""
        power_analyzer_day = self.power_analyzer_day()
        power_analyzer_day = power_analyzer_day.drop(['Индикатор'], axis=1)

        # Сравниваем мощности превышения. Если утро больше вечера то 0, если нет, записываем разницу
        def rule(x1, x2):
            return 0 if x1 > x2 else round(x2 - x1, 1)

        power_analyzer_day['Мощность превышения вечернего максимума утренним, кВт'] = \
            power_analyzer_day.apply(
                lambda x: rule(x['Максимум активной мощности в часы утреннего максимума энергосистемы , кВт'],
                               x['Максимум активной мощности в часы вечернего максимума энергосистемы , кВт']), axis=1)

        # Сравнение утреннего максимума и вечернего
        power_analyzer_day['Индикатор'] = power_analyzer_day.iloc[:, 2] > power_analyzer_day.iloc[:, 1]

        return power_analyzer_day

    def dd_power_analyzer_month(self):
        """ Индекс - период
        Индикатор - количество дней в месяце превышение вечер утренним пиков',
        'Сред мощность превышения вечернего максимума утренним, кВт',
       'Макс мощность превышения вечернего максимума утренним, кВт',
       'Выполняется условие оплаты по ДД-тарифу?'
       'Наибольший максимум активной мощности в часы утреннего максимума '
                                                   'энергосистемы , кВт'
       'Наибольший максимум активной мощности в часы вечернего максимума '
                                                   'энергосистемы , кВт'"""
        dd_power_analyzer_day = self.dd_power_analyzer_day().reset_index()
        # Создаем дополнительные для анализа столбцы
        dd_power_analyzer_day['Сред мощность превышения вечернего максимума утренним, кВт'] = \
            dd_power_analyzer_day['Мощность превышения вечернего максимума утренним, кВт']
        dd_power_analyzer_day['Макс мощность превышения вечернего максимума утренним, кВт'] = \
            dd_power_analyzer_day['Мощность превышения вечернего максимума утренним, кВт']
        dd_power_analyzer_day['Наибольший максимум активной мощности в часы утреннего максимума энергосистемы, кВт'] = \
            dd_power_analyzer_day['Максимум активной мощности в часы утреннего максимума энергосистемы , кВт']
        dd_power_analyzer_day['Наибольший максимум активной мощности в часы вечернего максимума энергосистемы, кВт'] = \
            dd_power_analyzer_day['Максимум активной мощности в часы вечернего максимума энергосистемы , кВт']

        dd_power_analyzer_day = dd_power_analyzer_day.drop(['Мощность превышения вечернего максимума утренним, кВт'],
                                                           axis=1)


        dd_power_analyzer_month = round(dd_power_analyzer_day.groupby('Период наблюдений'). \
                                        aggregate({'Индикатор': 'sum',
                                                   'Сред мощность превышения вечернего максимума утренним, кВт': 'mean',
                                                   'Макс мощность превышения вечернего максимума утренним, кВт': 'max',
                                                   'Наибольший максимум активной мощности в часы утреннего максимума '
                                                   'энергосистемы, кВт': 'max',
                                                   'Наибольший максимум активной мощности в часы вечернего максимума '
                                                   'энергосистемы, кВт': 'max',
                                                   }), 2)


        # Правило для выводов о Тарифе либо Д либо ДД, в зависимости от того ,превышает ли вечерний
        # максимум утренний в расчетной периоде
        def rule_temp(x,y):
            return 'Да' if x > y else 'Нет'

        # # Правило для выводов о Тарифе либо Д либо ДД, в зависимости от того ,превышает ли вечерний
        # # максимум утренний в расчетной периоде
        # def rule(x):
        #     return 'Да' if x == 0 else 'Нет'

        dd_power_analyzer_month['Выполняется условие оплаты по ДД-тарифу?'] = dd_power_analyzer_month. \
            apply(lambda x: rule_temp(x.iloc[3],x.iloc[4]), axis=1)

        return dd_power_analyzer_month

    def calculation(self, type_tariff=0):
        """key - ключ расчета:
        0 - считает по условию как получается в действительности, т.е, если выполняется условие
        расчета ДД - будет считать как ДД, если нет - осуществляется переход к расчету Д тарифа,
        1 -расчет выполняется исключительно по условию ДД-тарифа"""
        # Анализ по дням без учета условия превышения вечер утром
        energy_analyzer_day = self.dd_energy_analyzer_day().reset_index().set_index('Период наблюдений')
        tariff_coefficients = self.calculation_tariff_coefficients()
        power_analyzer_day = self.dd_power_analyzer_day().reset_index().set_index('Период наблюдений')
        power_analyzer_month = self.power_analyzer_month().iloc[:, [3]]

        # Вытягиваем информацию о количество исследованных дней
        number_of_days_examined = self.energy_analyzer_day().reset_index().set_index('Период наблюдений').iloc[:, [2]]
        merge = pd.concat([energy_analyzer_day, power_analyzer_day, tariff_coefficients, power_analyzer_month], axis=1)
        merge['Тариф на электрическую мощность, руб/кВт'] = self.tariff_power
        merge['Тариф на электрическую энергию, руб/кВтч'] = self.tariff_energy
        condition_pay = self.dd_power_analyzer_month().iloc[:, [5]]

        condition_pay = condition_pay.replace('Нет', int(0))
        condition_pay = condition_pay.replace('Да', int(1))

        merge = pd.concat([merge, number_of_days_examined, condition_pay], axis=1)

        df_day = merge.iloc[:, [0, 12, 18, 13, 17, 19, 14, 1, 16, 2, 15, 3, 4, 7, 8, 21, 20]]

        df_day = df_day.fillna(0)

        # Переделываем индикатор. Если в месяце есть дни, где пик вечера превышает утро, то по месяцу 1, нет  - 0
        temp = df_day.reset_index().groupby(['Период наблюдений']).aggregate({'Выполняется условие оплаты по ДД-тарифу?':
                                                                                  'mean'})

        df_day = df_day.drop(['Выполняется условие оплаты по ДД-тарифу?'], axis=1) # 1 - выполняется, 0 - нет
        df_day = pd.concat([df_day, temp], axis=1)

        # Расчет суточной за мощность
        def rule_power(key, ind, tariff_power, ka, pw, number_of_days):
            if key == 0:  # 0 - ДД или Д тариф от условия, 1 - исключительно ДД-тариф
                if ind == 1:  # Проверяет есть ли дни где максимум вечера больше утра, если нет - ДД-тариф
                    return round(tariff_power * ka * pw / number_of_days)
                else:
                    return round(tariff_power * pw / number_of_days)
            else:
                return round(tariff_power * ka * pw / number_of_days)


        df_day['Оплата за электрическую мощность, руб'] = \
            df_day.apply(lambda x: rule_power(key=type_tariff,
                                              ind=x['Выполняется условие оплаты по ДД-тарифу?'],
                                              tariff_power=x['Тариф на электрическую мощность, руб/кВт'],
                                              ka=x['Понижающий коэффициент к основной ставке'],
                                              pw=x['Максимум активной мощности в часы максимума '
                                                   'энергосистемы, кВт'],
                                              number_of_days=x['Количество дней в месяце']), axis=1)


        # Расчет суточной оплаты за электроэнергию в ночной зоне
        def rule_ener_night(key, ind, tariff_energy, kn, wn):
            if key == 0:  # 0 - ДД или Д тариф от условия, 1 - исключительно ДД-тариф
                if ind == 1:  # Проверяет есть ли дни где максимум вечера больше утра, если нет - ДД-тариф
                    return round(tariff_energy * kn * wn)
                else:
                    return 0
            else:
                return round(tariff_energy * kn * wn)


        df_day['Оплата за электроэнергию в ночной зоне, руб'] = \
            df_day.apply(lambda x: rule_ener_night(key=type_tariff,
                                                   ind=x['Выполняется условие оплаты по ДД-тарифу?'],
                                                   tariff_energy=x['Тариф на электрическую энергию, руб/кВтч'],
                                                   kn=x['Тарифный ночной коэффициент'],
                                                   wn=x['Расход электроэнергии в ночной зоне, кВтч']), axis=1)
        print(list(df_day))

        # Расчет суточной оплаты за электроэнергию в пиковой зоне
        def rule_ener_peak(key, ind, tariff_energy, kp, wp):
            if key == 0:  # 0 - ДД или Д тариф от условия, 1 - исключительно ДД-тариф
                if ind == 1:  # Проверяет есть ли дни где максимум вечера больше утра, если нет - ДД-тариф
                    return round(tariff_energy * kp * wp)
                else:
                    return 0
            else:
                return round(tariff_energy * kp * wp)

        df_day['Оплата за электроэнергию в пиковой зоне, руб'] = \
            df_day.apply(lambda x: rule_ener_peak(key=type_tariff,
                                                  ind=x['Выполняется условие оплаты по ДД-тарифу?'],
                                                  tariff_energy=x['Тариф на электрическую энергию, руб/кВтч'],
                                                  kp=x['Тарифный пиковый коэффициент'],
                                                  wp=x['Расход электроэнергии в пиковой зоне, кВтч']), axis=1)

        # Расчет суточной оплаты за электроэнергию в полупиковой зоне
        def rule_ener_ppeak(key, ind, tariff_energy, kpp, wpp):  # расход ээ в полупиквой зоне
            if key == 0:  # 0 - ДД или Д тариф от условия, 1 - исключительно ДД-тариф
                if ind == 1:  # Проверяет есть ли дни где максимум вечера больше утра, если нет - ДД-тариф
                    return round(tariff_energy * kpp * wpp)
                else:
                    return 0
            else:
                return round(tariff_energy * kpp * wpp)

        df_day['Оплата за электроэнергию в полупиковой зоне, руб'] = \
            df_day.apply(lambda x: rule_ener_ppeak(key=type_tariff,
                                                   ind=x['Выполняется условие оплаты по ДД-тарифу?'],
                                                   tariff_energy=x['Тариф на электрическую энергию, руб/кВтч'],
                                                   kpp=x['Тарифный полупиковый коэффициент'],
                                                   wpp=x['Расход электроэнергии в полупиковой зоне, кВтч']), axis=1)

        def rule_full_energy(key, tariff_energy, pay_n, pay_p, pay_pp, wsum):  # суммарная оплата
            if key == 0:  # 0 - ДД или Д тариф от условия, 1 - исключительно ДД-тариф
                if (pay_n + pay_p + pay_pp) != 0:
                    return pay_n + pay_p + pay_pp
                else:
                    return round(tariff_energy * wsum)
            else:
                return pay_n + pay_p + pay_pp

        df_day['Оплата за электроэнергию, руб'] = \
            df_day.apply(
                lambda x: rule_full_energy(key=type_tariff, tariff_energy=x['Тариф на электрическую энергию, руб/кВтч'],
                                           pay_n=x['Оплата за электроэнергию в ночной зоне, руб'],
                                           pay_p=x['Оплата за электроэнергию в пиковой зоне, руб'],
                                           pay_pp=x['Оплата за электроэнергию в полупиковой зоне, руб'],
                                           wsum=x['Суммарный расход электроэнергии, кВтч']), axis=1)

        df_day['Суммарная оплата за электроэнергию и мощность, руб'] = \
            round(df_day['Оплата за электрическую мощность, руб'] + df_day['Оплата за электроэнергию, руб'])

        df_day['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] = \
           round( df_day['Суммарная оплата за электроэнергию и мощность, руб'] / \
            df_day['Суммарный расход электроэнергии, кВтч'],4)

        self.mean_tariff = round(df_day['Суммарная оплата за электроэнергию и мощность, руб'].sum() / \
                                 df_day['Суммарный расход электроэнергии, кВтч'].sum(), 4)

        self.df_pay_energy_day = df_day


        # Список столбцов суточного фрейма
        # 'День'
        # 'Количество дней в месяце
        # 'Основная ставка, руб / кВт',
        # 'Понижающий коэффициент к основной ставке',
        # 'Максимум активной мощности в часы максимума энергосистемы, кВт',
        # 'Дополнительная ставка, руб/кВтч', 'Тарифный ночной коэффициент',
        # 'Расход электроэнергии в ночной зоне, кВтч',
        # 'Тарифный пиковый коэффициент',
        # 'Расход электроэнергии в пиковой зоне, кВтч',
        # 'Тарифный полупиковый коэффициент',
        # 'Расход электроэнергии в полупиковой зоне, кВтч',
        # 'Суммарный расход электроэнергии, кВтч',
        # 'Максимум активной мощности в часы утреннего максимума энергосистемы , кВт',
        # 'Максимум активной мощности в часы вечернего максимума энергосистемы , кВт',
        # 'Количество исследованных дней',
        # 'Количество дней превышения вечерних пиков в месяце',
        # 'Оплата за электрическую мощность, руб',
        # 'Оплата за электроэнергию в ночной зоне, руб',
        # 'Оплата за электроэнергию в пиковой зоне, руб',
        # 'Оплата за электроэнергию в полупиковой зоне, руб',
        # 'Оплата за электроэнергию, руб',
        # 'Суммарная оплата за электроэнергию и мощность, руб'


        # ________________ ОПЛАТА ПО МЕСЯЦАМ_____________________
        df_month = df_day.groupby(['Период наблюдений']).aggregate(
            {'Количество дней в месяце': 'mean',
             'Количество исследованных дней': 'sum',
             'Тариф на электрическую мощность, руб/кВт': lambda x: np.round(np.mean(x), 4),
             'Понижающий коэффициент к основной ставке': 'mean',
             'Максимум активной мощности в часы максимума энергосистемы, кВт': lambda x: np.round(np.mean(x), 2),
             'Тариф на электрическую энергию, руб/кВтч': lambda x: np.round(np.mean(x), 4),
             'Тарифный ночной коэффициент': lambda x: np.round(np.mean(x), 4),
             'Расход электроэнергии в ночной зоне, кВтч': 'sum',
             'Тарифный пиковый коэффициент': lambda x: np.round(np.mean(x), 4),
             'Расход электроэнергии в пиковой зоне, кВтч': 'sum',
             'Тарифный полупиковый коэффициент': lambda x: np.round(np.mean(x), 2),
             'Расход электроэнергии в полупиковой зоне, кВтч': 'sum',
             'Выполняется условие оплаты по ДД-тарифу?': 'mean',
             'Оплата за электрическую мощность, руб': 'sum',
             'Оплата за электроэнергию в ночной зоне, руб': 'sum',
             'Оплата за электроэнергию в пиковой зоне, руб': 'sum',
             'Оплата за электроэнергию в полупиковой зоне, руб': 'sum',
             'Оплата за электроэнергию, руб': 'sum',
             'Суммарная оплата за электроэнергию и мощность, руб': 'sum',
             'Суммарный расход электроэнергии, кВтч': 'sum',
             })



        df_day['Средний тариф за 1 кВт·ч электроэнергии, руб/ кВт·ч'] = \
           round( df_day['Суммарная оплата за электроэнергию и мощность, руб'] / \
            df_day['Суммарный расход электроэнергии, кВтч'], 4)

        self.df_pay_energy_month = df_month
        self.sum_pay_power = float(round(self.df_pay_energy_month.iloc[:, [13]].sum()))
        self.sum_pay_energy_night = float(round(self.df_pay_energy_month.iloc[:, [14]].sum()))
        self.sum_pay_energy_peak = float(round(self.df_pay_energy_month.iloc[:, [15]].sum()))
        self.sum_pay_energy_half_peak = float(round(self.df_pay_energy_month.iloc[:, [16]].sum()))
        self.sum_pay_energy = float(round(self.df_pay_energy_month.iloc[:, [17]].sum()))
        self.sum_pay_power_and_energy = float(round(self.df_pay_energy_month.iloc[:, [18]].sum()))


    def write_to_exel_buffer(self):

        if self.df_pay_energy_day is not None:
            pass
        else:
            self.calculation()
        if self.df_pay_energy_month is not None:
            pass
        else:
            self.calculation()
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            self.dd_power_analyzer_day().reset_index().to_excel(writer, sheet_name='Сравнение утр и веч пика.xlsx',
                                                                    index=False)
            self.dd_power_analyzer_month().reset_index().to_excel(writer, sheet_name='Мес сравн пиков.xlsx',
                                                                      index=False)
            self.df_pay_energy_day.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по суткам.xlsx', index=False)
            self.df_pay_energy_month.reset_index().to_excel(writer, sheet_name='Плата за ЭЭ по месяцам.xlsx',
                                                            index=False)
        processed_data = buffer.getvalue()
        return processed_data


class CompareOfTariffs():
    """Сравнение тарифов между собой"""

    def __init__(self, df, ab, bb, kt, kb, declared):
        self.df = df
        self.ab = ab
        self.bb = bb
        self.kt = kt
        self.kb = kb
        self.declared = declared

    def price_per_one_energy(self):
        """Выводит сортированный результат сравнения тарифов"""
        # Расчет Д-тарифа с заявленной мощностью
        declared = DTariffDeclared(self.df, self.ab, self.bb, self.kt, self.kb, self.declared)
        declared.calculation()
        pay_declared_tariff = declared.df_pay_energy_month.iloc[:, [-1]]

        # Расчет Д-тарифа с фактической мощностью
        d_tariff = DTariff(self.df, self.ab, self.bb, self.kt, self.kb)
        d_tariff.calculation()
        pay_d_tariff = d_tariff.df_pay_energy_month.iloc[:, [-1]]

        # Расчет ДД-тарифа с фактической нагрузкой
        dif_tariff = DifferTariff(self.df, self.ab, self.bb, self.kt, self.kb)
        dif_tariff.calculation(type_tariff=0)
        pay_dif_tariff = dif_tariff.df_pay_energy_month.iloc[:, [-1]]

        # Расчет ДД-тарифа с регулированием нагрузкой
        dif_tariff_reg = DifferTariff(self.df, self.ab, self.bb, self.kt, self.kb)
        dif_tariff_reg.calculation(type_tariff=1)
        pay_dif_tariff_reg = dif_tariff_reg.df_pay_energy_month.iloc[:, [-1]]

        df_total = pd.concat([pay_d_tariff, pay_declared_tariff, pay_dif_tariff, pay_dif_tariff_reg], axis=1)

        dict = {'Д-тариф с заявленной мощностью': declared.sum_pay_power_and_energy,
                'Д-тариф с фактической мощностью': d_tariff.sum_pay_power_and_energy,
                'ДД-тариф с без изменения режима нагрузки': dif_tariff.sum_pay_power_and_energy,
                'ДД-тариф при регулировании нагрузкой': dif_tariff_reg.sum_pay_power_and_energy}
        list_key = sorted(dict.items(), key=lambda item: item[1])

        return list_key


def compare(energy, declared_pay, d_pay, dd_pay, dd_pay_regul):
    """Функция используется для создания отчета, рассчитывает средний тариф, на входе датафреймы: индекс - период,
    столбцы - электроэнергия и оплаты при разные тарифах"""
    df = pd.concat([energy, declared_pay], axis=1)
    df['Средняя плата ЭЭ Д-тарифа заявленной мощности, руб/кВт'] = \
        df.apply(lambda x: (round(x.iloc[1] / x.iloc[0], 3) if x.iloc[0] != 0 else 0), axis=1)
    df = pd.concat([df, d_pay], axis=1)
    df['Средняя плата ЭЭ Д-тарифа фактической мощности, руб/кВт'] = \
        df.apply(lambda x: (round(x.iloc[3] / x.iloc[0], 3) if x.iloc[0] != 0 else 0), axis=1)
    df = pd.concat([df, dd_pay], axis=1)
    df['Средняя плата ЭЭ ДД-тарифа без регулирования нагрузкой, руб/кВт'] = \
        df.apply(lambda x: (round(x.iloc[5] / x.iloc[0], 3) if x.iloc[0] != 0 else 0), axis=1)
    df = pd.concat([df, dd_pay_regul], axis=1)
    df['Средняя плата ЭЭ ДД-тарифа при регулирования нагрузкой, руб/кВт'] = \
        df.apply(lambda x: (round(x.iloc[7] / x.iloc[0], 3) if x.iloc[0] != 0 else 0), axis=1)
    return df


def roll_time(series, shift):
    """Функция смещает по кругу данные на шаг shift каждые 48 значений, т.е. каждый день,
    тем самым позволяет смещать производственный цикл
    Исходные данные series - пандосовская серия, shift - шаг смещения (1-на 30 минут; -1 - на минус 30 минут)"""
    count_day = series.count() // 48
    ser = pd.Series(dtype='float64')
    for d in range(count_day):
        end = 0
        temp_list = []
        for s in series[d * 48::]:
            if end != 48:
                temp_list.append(s)
                end += 1
            else:
                break
        list = np.roll(temp_list, shift)
        temp_series = pd.Series(np.squeeze(list))
        ser = ser.append(temp_series)
    result = ser
    return result


def fun_calc_roll(df_input_statistics, df_declared_power, ab, bb, kt, kb, shift_left=6, shift_right=6):
    """
    Функция с заданным шагом смещает рабочее время влево shift_left и вправо shift_right и просчитывает
    все варианты тарифов.
    dt_left - количество шагов смещения по 30 минут влево от базы, т.е. если dt_left = 6, значит 30*6=180/60=3 ч
    dt_right - количество шагов смещения по 30 минут вправо от базы
    Результат: датафрем индекс- шаг смещения, столбцы
    'Оплата за электрическую мощность, руб.\n (Д-тариф с заявленной мощностью)',
    'Оплата за электрическую энергию, руб \n (Д-тариф с заявленной мощностью)',
    'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с заявленной мощностью)',
    'Оплата за электрическую мощность, руб.\n (Д-тариф с фактической мощностью)',
    'Оплата за электрическую энергию, руб \n (Д-тариф с фактической мощностью)',
    'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с фактической мощностью)',
    'Оплата за электрическую мощность, руб.\n (ДД-тариф без регулирования)',
    'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф без регулирования)'
    'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф без регулирования)'
    'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф без регулирования)'
    'Оплата за электрическую энергию, руб \n (ДД-тариф без регулирования)'
    'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф без регулирования)'
    'Оплата за электрическую мощность, руб.\n (ДД-тариф с регулированием)'
     'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф с регулированием)'
    'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф с регулированием)'
    'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф с регулированием)'
    'Оплата за электрическую энергию, руб \n (ДД-тариф с регулированием)'
    'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф с регулированием)'
    """
    data = pd.DataFrame()
    df1 = df_input_statistics
    df3 = df_declared_power
    if shift_left <= 24 and shift_right <= 24:
        for dt in range(-abs(shift_left), shift_right + 1, 1):
            print('Шаг смещения производственного цикла - ', dt * 0.5, 'ч')
            df1.iloc[:, [1, 2]] = df1.iloc[:, [1, 2]].apply(lambda x: roll_time(series=x, shift=dt)).values

            d_tariff_declared = DTariffDeclared(df=df1, ab=ab, bb=bb, kt=kt, kb=kb, declared=df3)
            d_tariff_declared.calculation()
            d_tariff_declared_sum_pay_power = d_tariff_declared.sum_pay_power
            d_tariff_declared_sum_pay_energy = d_tariff_declared.sum_pay_energy
            d_tariff_declared_sum_pay_power_and_energy = d_tariff_declared.sum_pay_power_and_energy

            d_tariff = DTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
            d_tariff.calculation()
            d_tarif_sum_pay_power = d_tariff.sum_pay_power
            d_tarif_sum_pay_energy = d_tariff.sum_pay_energy
            d_tariff_sum_pay_power_and_energy = d_tariff.sum_pay_power_and_energy

            dif_tariff = DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
            dif_tariff.calculation(type_tariff=0)
            dif_tariff_sum_pay_power = dif_tariff.sum_pay_power
            dif_tariff_sum_pay_energy_night = dif_tariff.sum_pay_energy_night
            dif_tariff_sum_pay_energy_peak = dif_tariff.sum_pay_energy_peak
            dif_tariff_sum_pay_energy_half_peak = dif_tariff.sum_pay_energy_half_peak
            dif_tariff_sum_pay_energy = dif_tariff.sum_pay_energy
            dif_tariff_sum_pay_power_and_energy = dif_tariff.sum_pay_power_and_energy

            dif_tariff_reg = DifferTariff(df=df1, ab=ab, bb=bb, kt=kt, kb=kb)
            dif_tariff_reg.calculation(type_tariff=1)
            dif_tariff_reg_sum_pay_power = dif_tariff_reg.sum_pay_power
            dif_tariff_reg_sum_pay_energy_night = dif_tariff_reg.sum_pay_energy_night
            dif_tariff_reg_sum_pay_energy_peak = dif_tariff_reg.sum_pay_energy_peak
            dif_tariff_reg_sum_pay_energy_half_peak = dif_tariff_reg.sum_pay_energy_half_peak
            dif_tariff_reg_sum_pay_energy = dif_tariff_reg.sum_pay_energy
            dif_tariff_reg_sum_pay_power_and_energy = dif_tariff_reg.sum_pay_power_and_energy

            temp_data = pd.DataFrame({'Оплата за электрическую мощность, руб.\n (Д-тариф с заявленной мощностью)':
                                          [d_tariff_declared_sum_pay_power],
                                      'Оплата за электрическую энергию, руб \n (Д-тариф с заявленной мощностью)':
                                          [d_tariff_declared_sum_pay_energy],
                                      'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с заявленной мощностью)':
                                          [d_tariff_declared_sum_pay_power_and_energy],

                                      'Оплата за электрическую мощность, руб.\n (Д-тариф с фактической мощностью)':
                                          [d_tarif_sum_pay_power],
                                      'Оплата за электрическую энергию, руб \n (Д-тариф с фактической мощностью)':
                                          [d_tarif_sum_pay_energy],
                                      'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с фактической мощностью)':
                                          [d_tariff_sum_pay_power_and_energy],

                                      'Оплата за электрическую мощность, руб.\n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_power],
                                      'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_energy_night],
                                      'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_energy_peak],
                                      'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_energy_half_peak],
                                      'Оплата за электрическую энергию, руб \n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_energy],
                                      'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф без регулирования)':
                                          [dif_tariff_sum_pay_power_and_energy],

                                      'Оплата за электрическую мощность, руб.\n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_power],
                                      'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_energy_night],
                                      'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_energy_peak],
                                      'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_energy_half_peak],
                                      'Оплата за электрическую энергию, руб \n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_energy],
                                      'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф с регулированием)':
                                          [dif_tariff_reg_sum_pay_power_and_energy]
                                      }, index=[dt * 0.5])
            temp_data.index.name = 'Шаг смещения, ч'
            data = data.append(temp_data)
        return data
    else:
        return print('Шаг смещения влево или вправо не должен превышать значение 24 единицы, что составляет 12 ч')


class RollStepThread():
    """Реализована многопоточность! каждый поток считает разные шаг смещения"""

    def __init__(self, df_input_statistics, df_declared_power, ab, bb, kt, kb, step):
        super().__init__()

        self.df_input_statistics = df_input_statistics
        self.df_declared_power = df_declared_power
        self.step = step
        self.ab = ab
        self.bb = bb
        self.kt = kt
        self.kb = kb
        self.result = None

    def func(self):
        df1 = self.df_input_statistics
        df3 = self.df_declared_power

        d_tariff_declared = DTariffDeclared(df=df1, ab=self.ab, bb=self.bb, kt=self.kt, kb=self.kb, declared=df3)
        d_tariff_declared.calculation()
        d_tariff_declared_sum_pay_power = d_tariff_declared.sum_pay_power
        d_tariff_declared_sum_pay_energy = d_tariff_declared.sum_pay_energy
        d_tariff_declared_sum_pay_power_and_energy = d_tariff_declared.sum_pay_power_and_energy

        d_tariff = DTariff(df=df1, ab=self.ab, bb=self.bb, kt=self.kt, kb=self.kb)
        d_tariff.calculation()
        d_tarif_sum_pay_power = d_tariff.sum_pay_power
        d_tarif_sum_pay_energy = d_tariff.sum_pay_energy
        d_tariff_sum_pay_power_and_energy = d_tariff.sum_pay_power_and_energy

        dif_tariff = DifferTariff(df=df1, ab=self.ab, bb=self.bb, kt=self.kt, kb=self.kb)
        dif_tariff.calculation(type_tariff=0)
        dif_tariff_sum_pay_power = dif_tariff.sum_pay_power
        dif_tariff_sum_pay_energy_night = dif_tariff.sum_pay_energy_night
        dif_tariff_sum_pay_energy_peak = dif_tariff.sum_pay_energy_peak
        dif_tariff_sum_pay_energy_half_peak = dif_tariff.sum_pay_energy_half_peak
        dif_tariff_sum_pay_energy = dif_tariff.sum_pay_energy
        dif_tariff_sum_pay_power_and_energy = dif_tariff.sum_pay_power_and_energy

        dif_tariff_reg = DifferTariff(df=df1, ab=self.ab, bb=self.bb, kt=self.kt, kb=self.kb)
        dif_tariff_reg.calculation(type_tariff=1)
        dif_tariff_reg_sum_pay_power = dif_tariff_reg.sum_pay_power
        dif_tariff_reg_sum_pay_energy_night = dif_tariff_reg.sum_pay_energy_night
        dif_tariff_reg_sum_pay_energy_peak = dif_tariff_reg.sum_pay_energy_peak
        dif_tariff_reg_sum_pay_energy_half_peak = dif_tariff_reg.sum_pay_energy_half_peak
        dif_tariff_reg_sum_pay_energy = dif_tariff_reg.sum_pay_energy
        dif_tariff_reg_sum_pay_power_and_energy = dif_tariff_reg.sum_pay_power_and_energy

        temp_data = pd.DataFrame({'Шаг смещения производственного цикла, ч':
                                      [self.step],

                                  'Оплата за электрическую мощность, руб.\n (Д-тариф с заявленной мощностью)':
                                      [d_tariff_declared_sum_pay_power],
                                  'Оплата за электрическую энергию, руб \n (Д-тариф с заявленной мощностью)':
                                      [d_tariff_declared_sum_pay_energy],
                                  'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с заявленной мощностью)':
                                      [d_tariff_declared_sum_pay_power_and_energy],

                                  'Оплата за электрическую мощность, руб.\n (Д-тариф с фактической мощностью)':
                                      [d_tarif_sum_pay_power],
                                  'Оплата за электрическую энергию, руб \n (Д-тариф с фактической мощностью)':
                                      [d_tarif_sum_pay_energy],
                                  'Суммарная оплата за электроэнергию и мощность, руб.\n (Д-тариф с фактической мощностью)':
                                      [d_tariff_sum_pay_power_and_energy],

                                  'Оплата за электрическую мощность, руб.\n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_power],
                                  'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_energy_night],
                                  'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_energy_peak],
                                  'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_energy_half_peak],
                                  'Оплата за электрическую энергию, руб \n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_energy],
                                  'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф без регулирования)':
                                      [dif_tariff_sum_pay_power_and_energy],

                                  'Оплата за электрическую мощность, руб.\n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_power],
                                  'Оплата за электрическую энергию в ночной зоне, руб \n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_energy_night],
                                  'Оплата за электрическую энергию в пиковой зоне, руб \n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_energy_peak],
                                  'Оплата за электрическую энергию в полупиковой зоне, руб \n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_energy_half_peak],
                                  'Оплата за электрическую энергию, руб \n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_energy],
                                  'Суммарная оплата за электроэнергию и мощность, руб.\n (ДД-тариф с регулированием)':
                                      [dif_tariff_reg_sum_pay_power_and_energy]
                                  })
        global global_result
        global_result = global_result.append(temp_data)

    def run(self):
        self.func()


def step_thread(ab, bb, kt, kb, df_input_statistics, df_declared_power, shift_left=6, shift_right=6):
    df1 = df_input_statistics
    if shift_left <= 24 and shift_right <= 24:
        list_df = []

        def roll(df, dt):
            df_new = df.copy()
            df_new.iloc[:, [1, 2]] = df_new.iloc[:, [1, 2]].apply(lambda x: roll_time(series=x, shift=dt)).values
            return df_new

        step_list = []
        for dt in range(-abs(shift_left), shift_right + 1, 1):
            print('Шаг смещения производственного цикла - ', dt * 0.5, 'ч')
            try:
                new_df = roll(df=df1, dt=dt)
                list_df.append(new_df)
                step_list.append(dt * 0.5)
            except ValueError:
                print('Не все исходные данные содержат 48 значений мощности в сутках.')
                sys.exit()

        pars_list = []
        for i, df in enumerate(list_df):
            step = step_list[i]
            pars = RollStepThread(df_input_statistics=df, df_declared_power=df_declared_power, ab=ab, bb=bb,
                                  kt=kt, kb=kb, step = step)
            pars_list.append(pars)

        for pars in pars_list:
            pars.run()


        global global_result
        global_result = global_result.set_index('Шаг смещения производственного цикла, ч')
        global_result.sort_index(inplace=True)
        global_result.to_excel('Result_Excel/Оплата ЭЭ при смещении цикла производства.xlsx',
                               sheet_name = 'Резуальтат', index = False)
        return global_result

    else:
        print('Шаг смещения влево или вправо не должен превышать значение 24 единицы, что составляет 12 ч')
