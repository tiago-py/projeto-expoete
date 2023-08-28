from cProfile import label
from distutils.command.config import config
from re import template
from turtle import color
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash_bootstrap_templates import ThemeSwitchAIO


# ========= App ============== #
FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc_css])
app.scripts.config.serve_locally = True
server = app.server

# ========== Styles ============ #
template_theme1 = "spacelab"
template_theme2 = "darkly"
url_theme1 = dbc.themes.SPACELAB
url_theme2 = dbc.themes.DARKLY
tab_card = {'height': '100%'}


# ===== Reading n cleaning File ====== #
df_main = pd.read_csv("data_gas.csv")
main_config={
    "hovermode": "x unified",
    "legend": {"yanchor":"top",
            "y":0.9,
            "xanchor":"left",
            "x":0.1,
            "title":{"text":None},
            "font":{"color":"white"},
            "bgcolor":"rgba(0,0,0,0.5)"},
    "margin":{"l":0, "r":0, "t":10, "b":0}
}

# PASSAR PARA DATE TIME
df_main['DATA INICIAL'] = pd.to_datetime(df_main["DATA INICIAL"])
df_main['DATA FINAL'] = pd.to_datetime(df_main['DATA FINAL'])
# CALCULAR A MÉDIA
df_main['DATA MEDIA'] = (
    (df_main['DATA FINAL'] - df_main['DATA INICIAL'])/2) + df_main['DATA INICIAL']
# ORDENAR PELA MÉDIA
df_main = df_main.sort_values(by='DATA MEDIA', ascending=True)
# RENOMEAR COLUNAS
df_main.rename(columns={'DATA MEDIA': 'DATA'}, inplace=True)
df_main.rename(
    columns={'PREÇO MÉDIO REVENDA': 'VALOR REVENDA (R$/L)'}, inplace=True)

# CRIANDO COLUNA DE ANO
df_main["ANO"] = df_main["DATA"].apply(lambda x: str(x.year))
# PEGAR APENAS OS DADOS DA GASOLINA
df_main = df_main[df_main.PRODUTO == "GASOLINA COMUM"]

# RESET DATAFRAME
df_main = df_main.reset_index()

# EXCLUINDO COLUNAS
df_main.drop(['UNIDADE DE MEDIDA', 'COEF DE VARIAÇÃO REVENDA', 'COEF DE VARIAÇÃO DISTRIBUIÇÃO',
              'NÚMERO DE POSTOS PESQUISADOS', 'DATA INICIAL', 'DATA FINAL', 'PREÇO MÁXIMO DISTRIBUIÇÃO', 'PREÇO MÍNIMO DISTRIBUIÇÃO',
              'DESVIO PADRÃO DISTRIBUIÇÃO', 'MARGEM MÉDIA REVENDA', 'PREÇO MÍNIMO REVENDA', 'PREÇO MÁXIMO REVENDA', 'DESVIO PADRÃO REVENDA',
              'PRODUTO', 'PREÇO MÉDIO DISTRIBUIÇÃO'], inplace=True, axis=1)

# Para salvar no dcc.store
df_store = df_main.to_dict()

# =========  Layout  =========== #
app.layout = dbc.Container(children=[

    dcc.Store(id='dataset', data=df_store),
    dcc.Store(id='dataset_fixed', data=df_store),
    #DICIONARIO COM UMA CHAVE E UM VALOR 
    dcc.Store(id='controller', data={'play':False}),


    # LAYOUT
    # ROW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                     dbc.Col([
                         html.Legend("Analise dos preços da Gasolina")
                     ], sm=8),
                     dbc.Col([
                         html.I(className='fa fa-filter',
                                style={'font-size': '300%'})
                     ], sm=4, align="center")
                     ]),
                    dbc.Row([
                        dbc.Col([
                            ThemeSwitchAIO(aio_id='theme', themes=[
                                url_theme1, url_theme2]),
                        ])
                    ], style={'margin-top': '10px'}),
                    dbc.Row([
                        dbc.Col(
                             dbc.Button(
            ["Visite  ", html.I(className="fab fa-github",  style={'font-size': '100%'})],
            href="https://github.com/tiago-py", target="_blank",
            className="m-1",
        ),
                        )
                    ], style={'margin-top': '10px'})
                ])
            ], style=tab_card)
        ], sm=4, lg=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                     dbc.Col([
                         html.H3('Máximos e Mínimos'),
                         dcc.Graph(
                             id='static-maxmin', config={"displayModeBar": False, "showTips": False})
                     ])
                     ])
                ])
            ], style=tab_card)
        ], sm=8, lg=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                     dbc.Col([
                         html.H6('Ano de análise:'),
                         dcc.Dropdown(
                             id="select_ano",
                             value=df_main.at[df_main.index[1], 'ANO'],
                             clearable=False,
                             className='dbc',
                             options=[
                                {"label": x, "value": x} for x in df_main.ANO.unique()
                             ]),
                     ], sm=6),
                     dbc.Col([
                         html.H6('Região de análise'),
                         dcc.Dropdown(
                             id="select_regiao",
                             value=df_main.at[df_main.index[1], 'REGIÃO'],
                             clearable=False,
                             className='dbc',
                             options=[{"label": x, "value": x} for x in df_main.REGIÃO.unique()
                                      ]),
                     ], sm=6)
                     ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='regiaobar_graph', config={"displayModeBar": False, "showTips": False})
                        ], sm=12, md=6),
                        dbc.Col([
                            dcc.Graph(id='estadobar_graph', config={ "displayModeBar": False, "showTips": False})
                        ], sm=12, md=6)
                    ], style= {'column-gap':'0px'})
                ])
            ],style=tab_card)
        ],sm=12, lg=7)
    ],className='g-2 my-auto'),
  #ROW 2
  dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H3('Preço x Estado'),
                html.H6('Comparação temporal entre estados'),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="select_estados0",
                            value=[df_main.at[df_main.index[3],'ESTADO'],df_main.at[df_main.index[13],'ESTADO'],df_main.at[df_main.index[6],'ESTADO']],
                            clearable=False,
                            className='dbc',
                            multi= True,
                            options=[
                                {"label":x , "value":x} for x in df_main.ESTADO.unique()
                            ]),
                    ],sm=10),
                ]),
                dbc.Row(
                    dbc.Col([
                        dcc.Graph(id='animation_graph', config={"displayModeBar":False, "showTips":False})
                    ])
                )
            ])
        ],style= tab_card)
    ], sm=12, md=6, lg=5),
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H3('Comparação Direta'),
                html.H6('Qual preço é menor em um dado período de tempo?'),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="select_estado1",
                            value=df_main.at[df_main.index[3],'ESTADO'],
                            clearable=False,
                            className='dbc',
                            options=[
                                {"label":x , "value":x} for x in df_main.ESTADO.unique() 
                        ]   ),
                    ],sm=10,md=5),
                    dbc.Col([
                          dcc.Dropdown(
                            id="select_estado2",
                            value=df_main.at[df_main.index[1],'ESTADO'],
                            clearable=False,
                            className='dbc',
                            options=[
                                {"label":x , "value":x} for x in df_main.ESTADO.unique() 
                             ] ),  
                    ],sm=10, md=6)
                ],style={'margin-top':'20px'},justify='center'),
                dcc.Graph(id='direct_comparison_graph',config={"displayModeBar":False, "showTips":False}),
                html.P(id='desc_comparison', style={'color':'gray', 'font-size': '80% '}),
            ])
        ], style=tab_card)
    ], sm=12,md=6,lg=4),
    dbc.Col([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id= 'card1_indicators', config={"displayModeBar":False,"showTips":False},style={'margin-top':'30px'})
                    ])
                ], style=tab_card)
            ])
        ], justify='center',style={'padding-bottom':'7px', 'height':'50%'}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id= 'card2_indicators', config={"displayModeBar":False,"showTips":False},style={'margin-top':'30px'})
                    ])
                ], style=tab_card)
            ])
        ], justify='center',style={'height':'50%'}),
    ],sm=12, lg=3, style={'height':'100%'})
  ],className='g-2 my-auto'),

  #row 3
  dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.Row([
                dbc.Col([
                    dbc.Button([html.I(className='far fa-play-circle')], id="play-button", style={'margin-right':'15px'}),
                    dbc.Button([html.I(className= 'far fa-pause-circle')], id="stop-button")
                ],sm=12,md=1, style={'justify-content':'center', 'margin-top':'10px'}),
                dbc.Col([
                    dcc.RangeSlider(
                         id='rangeslider',
                         marks= {int(x): f'{x}' for x in df_main['ANO'].unique()},
                         step=3,
                         min=2004,
                         max=2021,
                         className='dbc',
                         value=[2004,2021],
                         dots=True,
                         pushable=3,
                         tooltip={'always_visible':False, 'placement': 'bottom'},

                    )  
                ],sm=12,md=10,style={'margin-top':'15px'}),
                #UPDATE NA ANIMAÇÃO 
                dcc.Interval(id='interval', interval=2000),
            ],className='g-1', style={'height':'20%', 'justify-content':'center'})
        ])
    ],className='g-2 my-auto')
  ])


], fluid=True, style={'height': '100%'})


# ======== Callbacks ========== #

#GRAFICO MAX/MIN
@app.callback(
    Output('static-maxmin', 'figure'),
    [Input('dataset', 'data'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def func(data, toggle):
    template = template_theme1 if toggle else template_theme2

    dff=pd.DataFrame(data)
    max= dff.groupby(['ANO'])['VALOR REVENDA (R$/L)'].max()
    min= dff.groupby(['ANO'])['VALOR REVENDA (R$/L)'].min()

    final_df = pd.concat([max, min], axis=1)
    final_df.columns = ['Max', 'Min']

    fig = px.line(final_df, x=final_df.index, y=final_df.columns, template=template)
    
    # UPDATE
    fig.update_layout(main_config, height=150, xaxis_title=None, yaxis_title=None)

    return fig
# CALBACK DAS BARRAS HORIZONTAIS 
@app.callback(
    [Output('regiaobar_graph', 'figure'),
    Output('estadobar_graph', 'figure')],
    [Input('dataset_fixed', 'data'),
    Input('select_ano', 'value'),
    Input('select_regiao', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)

# ALTERAÇÃO DO TEMA NO GRAFICO
def graph1(data, ano, regiao, toggle):
    template = template_theme1 if toggle else template_theme2

#FILTRANDO DATAFRAME POR ANO 
    df = pd.DataFrame(data)
    df_filtered = df[df.ANO.isin([ano])]

#CRIANDO UM DATEFRAME PARA REGIÃO/ ESTADO PARA POSSIBILITAR A MANIPULAÇÃO E A MÉDIA
    dff_regiao = df_filtered.groupby(['ANO', 'REGIÃO'])['VALOR REVENDA (R$/L)'].mean().reset_index()
    dff_estado = df_filtered.groupby(['ANO', 'ESTADO', 'REGIÃO'])['VALOR REVENDA (R$/L)'].mean().reset_index()
    dff_estado = dff_estado[dff_estado.REGIÃO.isin([regiao])]

#ORGANIZANDO DE FORMA ASCENDENTE
    dff_regiao = dff_regiao.sort_values(by='VALOR REVENDA (R$/L)',ascending=True)
    dff_estado = dff_estado.sort_values(by='VALOR REVENDA (R$/L)',ascending=True)

# DEFININDO 2 CASAS DECIMAIS 
    dff_regiao['VALOR REVENDA (R$/L)'] = dff_regiao['VALOR REVENDA (R$/L)'].round(decimals = 2)
    dff_estado['VALOR REVENDA (R$/L)'] = dff_estado['VALOR REVENDA (R$/L)'].round(decimals = 2)
#TEXTO PARA FIGURA 1 E PARA FIGURA 2 
    fig1_text = [f'{x} - R${y}' for x,y in zip(dff_regiao.REGIÃO.unique(), dff_regiao['VALOR REVENDA (R$/L)'].unique())]
    fig2_text = [f'R${y} - {x}' for x,y in zip(dff_estado.ESTADO.unique(), dff_estado['VALOR REVENDA (R$/L)'].unique())]

#GERANDO A FIGURA 1
    fig1 = go.Figure(go.Bar(
        x=dff_regiao['VALOR REVENDA (R$/L)'],
        y=dff_regiao['REGIÃO'],
        orientation='h',
        text=fig1_text,
        textposition='auto',
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12)
    ))
    #GERANDO A FIGURA 2
    fig2 = go.Figure(go.Bar(
        x=dff_estado['VALOR REVENDA (R$/L)'],
        y=dff_estado['ESTADO'],
        orientation='h',
        text=fig2_text,
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12) 
    ))

    #INSERINDO TEMPLATE
    fig1.update_layout(main_config, yaxis={'showticklabels':False}, height=150, template=template)
    fig2.update_layout(main_config, yaxis={'showticklabels':False}, height=150, template=template)

    # RANGE
    fig1.update_layout(xaxis_range=[dff_regiao['VALOR REVENDA (R$/L)'].max(), dff_regiao['VALOR REVENDA (R$/L)'].min() - 0.15])
    fig2.update_layout(xaxis_range=[dff_estado['VALOR REVENDA (R$/L)'].min() - 0.15, dff_estado['VALOR REVENDA (R$/L)'].max()])

    return [fig1, fig2]

# COMPARAÇÃO GRAFICO
@app.callback(
    Output('animation_graph', 'figure'),
    [Input('dataset', 'data'), 
    Input('select_estados0', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def animation(data, estados, toggle):

    template = template_theme1 if toggle else template_theme2
    

    dff = pd.DataFrame(data)
    #TODO NOME ESTIVER NA LINHA DE ESTADO ELE IRA TRAZER
    mask = dff.ESTADO.isin(estados)
    #TRAZENDO A FIG
    fig = px.line(dff[mask], x='DATA', y='VALOR REVENDA (R$/L)',
        color='ESTADO', template=template)
    
    # UPDATE
    fig.update_layout(main_config, height=400, xaxis_title=None)

    return fig

# COMPARAÇÃO DIRETA GRAFICO
@app.callback(
    [Output('direct_comparison_graph', 'figure'),
    Output('desc_comparison', 'children')],
    [Input('dataset', 'data'),
    Input('select_estado1', 'value'),
    Input('select_estado2', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def func(data, est1, est2, toggle):

    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    #SEPARANDO EM 2 DATAFRAME E CRIANDO UM FINAL 
    df1 = dff[dff.ESTADO.isin([est1])]
    df2 = dff[dff.ESTADO.isin([est2])]
    df_final = pd.DataFrame()
    
    #CRIANDO UMA VARIAVEL APARTIR DA MÉDIA DE VALORES DELES
    df_estado1 = df1.groupby(pd.PeriodIndex(df1['DATA'], freq="M"))['VALOR REVENDA (R$/L)'].mean().reset_index()
    df_estado2 = df2.groupby(pd.PeriodIndex(df2['DATA'], freq="M"))['VALOR REVENDA (R$/L)'].mean().reset_index()

    # DADO DE PERIODO DO PANDAS 
    df_estado1['DATA'] = pd.PeriodIndex(df_estado1['DATA'], freq="M")
    df_estado2['DATA'] = pd.PeriodIndex(df_estado2['DATA'], freq="M")

    #CRIANDO COLUNA DATA E A COLUNA DE VALOR DE REVENDA 
    df_final['DATA'] = df_estado1['DATA'].astype('datetime64[ns]')
    df_final['VALOR REVENDA (R$/L)'] = df_estado1['VALOR REVENDA (R$/L)']-df_estado2['VALOR REVENDA (R$/L)']
    
    
    fig = go.Figure()
    #TODO O CAMINHO PRIMEIRA LINHA
    fig.add_scattergl(name=est1, x=df_final['DATA'], y=df_final['VALOR REVENDA (R$/L)'])
    # ABAIXO DE ZERO
    fig.add_scattergl(name=est2, x=df_final['DATA'], y=df_final['VALOR REVENDA (R$/L)'].where(df_final['VALOR REVENDA (R$/L)'] > 0.00000))

    # UPDATE
    fig.update_layout(main_config, height=350, template=template)

    fig.update_yaxes(range = [-0.7,0.7])

    # MOSTRAR QUEM É MAIS BARATO EM ANOTAÇAÕ
    fig.add_annotation(text=f'{est2} é mais barato',
        xref="paper", yref="paper",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
            ),
        align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
        x=0.1, y=0.75, showarrow=False)

    fig.add_annotation(text=f'{est1} é mais barato',
        xref="paper", yref="paper",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
            ),
        align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
        x=0.1, y=0.25, showarrow=False) 

    # Definindo o texto
    text = f"Comparando {est1} e {est2}. Se a linha estiver acima do eixo X, {est2} tinha menor preço, do contrário, {est1} tinha um valor inferior"
    return [fig, text]

# INDICAFOR 1
@app.callback(
    Output("card1_indicators", "figure"),
    [Input('dataset', 'data'), 
    Input('select_estado1', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def card1(data, estado, toggle):

    template = template_theme1 if toggle else template_theme2


    dff = pd.DataFrame(data)
    #FILTRANDO POR ESTADOS
    df_final = dff[dff.ESTADO.isin([estado])]
    #ANO MIN E ANO MAX
    data1 = str(int(dff.ANO.min()) - 1)
    data2 = dff.ANO.max()   
    
    fig = go.Figure()
    #ADICIONAR  UM TRACE 
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:50%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        #ULTIMO VALOR REGISTRADO
        value = df_final.at[df_final.index[-1],'VALOR REVENDA (R$/L)'],
        #REFATORANDO O VALOR 
        number = {'prefix': "R$", 'valueformat': '.2f'},
        #DIFERENÇA
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'VALOR REVENDA (R$/L)']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

# INDICADOR 2
@app.callback(
    Output("card2_indicators", "figure"),
    [Input('dataset', 'data'), 
    Input('select_estado2', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def card2(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    #FILTRANDO POR ESTADOS
    df_final = dff[dff.ESTADO.isin([estado])]
    #ANO MIN E ANO MAX
    data1 = str(int(dff.ANO.min()) - 1)
    data2 = dff.ANO.max()
    
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number+delta",
       
        title = {"text": f"<span style='size:50%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        #ULTIMO VALOR REGISTRADO
        value = df_final.at[df_final.index[-1],'VALOR REVENDA (R$/L)'],
        #REFATORANDO O VALOR 
        number = {'prefix': "R$", 'valueformat': '.2f'},
        #DIFERENÇA
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'VALOR REVENDA (R$/L)']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

# DATA RANGE
@app.callback(
    Output('dataset', 'data'),
    [Input('rangeslider', 'value'),
    Input('dataset_fixed', 'data')], prevent_initial_call=True
)
def range_slider(range, data):
    dff = pd.DataFrame(data)
    #O DATAFRAME ESTA DENTRO DA DATA INICIAL E ATÉ A DATA FINAL 
    dff = dff[(dff['ANO'] >= f'{range[0]}-01-01') & (dff['ANO'] <= f'{range[1]}-31-12')]
    #TRANSFORMANDO EM DICIONARIO 
    data = dff.to_dict()
    # RETORNANDO DE VOLTA PARA O DATASET - PEGA DADOS DO FIXO E PASSA PARA O ALTERAVEL 
    return data

#  ANIMAÇÃO RANGESLIDER
@app.callback(
    Output('rangeslider', 'value'),
    Output('controller', 'data'), 

    Input('interval', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),
#ONDE O RANGESLIDER ESTÁ E SE ESTA PAUSADO OU RODANDO O PLAY
    State('rangeslider', 'value'), 
    State('controller', 'data'), 
    prevent_initial_callbacks = True)

def controller(n_intervals, play, stop, rangeslider, controller):
    #Qual dos inputs estão sendo clicados (play, pause)
    trigg = dash.callback_context.triggered[0]["prop_id"]

    # se for falso ele vira true ao clicar no play o rangeslider vai para 2007
    if ('play-button' in trigg and not controller["play"]):
        if not controller["play"]:
            controller["play"] = True
            rangeslider[1] = 2007
    # se o play estiver ativado ele pausa    
    elif 'stop-button' in trigg:
        if controller["play"]:
            controller["play"] = False
    # se ele chegar ao valor maximo ele pausa 
    if controller["play"]:
        if rangeslider[1] == 2021:
            controller['play'] = False
        rangeslider[1] += 1 if rangeslider[1] < 2021 else 0
    
    #retorna o valor da direita e se é para rodar ou não 
    return rangeslider, controller
# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
