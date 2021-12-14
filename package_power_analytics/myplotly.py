import plotly.graph_objs as go


def myplotly(dfx, *args, rolling=True, window_rolling=336):
    fig = go.Figure()
    for y in args:
        ser_x = dfx.iloc[:, 0]
        ser_y = y.iloc[:, 0]
        fig.add_trace(go.Scatter(x=ser_x, y=ser_y,
                                 opacity=0.9,
                                 mode='markers',
                                 marker=dict(color='#088A08', size=1, symbol='circle-open'),
                                 name='Исходные данные'))


        if (rolling == True) and (window_rolling<len(y)):
            y_roll = ser_y.rolling(window=window_rolling, center=True).mean()
            fig.add_trace(go.Scatter(x=ser_x, y=y_roll,
                                     opacity=1,
                                     mode='lines',
                                     marker=dict(color='#F60D0D', size=2, symbol='circle-open'),
                                     name='Сглаженные данные скользящей средней'))

        fig.update_layout(legend_orientation="h",
                          legend=dict(x=.5, xanchor="center"),
                          xaxis_title=dfx.columns[0],
                          yaxis_title=y.columns[0],
                          margin=dict(l=0, r=40, t=0, b=0),
                          # autosize = True,
                          # height=600,
                          # width=1000
                        )
    return fig


