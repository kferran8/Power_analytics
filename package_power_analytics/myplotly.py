import plotly.graph_objs as go

def myplotly(dfx, *dfy_list):
    fig = go.Figure()

    for y in dfy_list:
        fig.add_trace(go.Scatter(x=dfx, y=y,
                                        opacity=0.9,
                                        mode='markers',
                                        marker=dict(color='#088A08', size=1, symbol = 'circle-open'),
                                        name='Исходные данные'))





    # fig.add_trace(go.Scatter(x=df_new.iloc[:, 0], y=df_new.iloc[:, 3],
    #                                                   mode='markers',
    #                                                   marker=dict(color='#000000', size=marker_size_2,
    #                                                               symbol = 'square'),
    #                                                   name='Результаты кластеризации'))
    # fig.update_layout(legend_orientation="h",
    #                   legend=dict(x=.5, xanchor="center"),
    #                   title="Plot Title",
    #                   xaxis_title=df_new.columns[0],
    #                   yaxis_title=df_new.columns[2],
    #                   margin=dict(l=20, r=0, t=0, b=0),
    #                   # height=600,
    #                   # width=1000
    #                   )

    return fig










    return fig