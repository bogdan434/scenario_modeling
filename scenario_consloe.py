import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, dash_table
from dash import html
from dash.dcc import send_data_frame
from dash.dependencies import Input, Output, State
from app_functions import *

app = dash.Dash()


data = load_data()
data['Year'] = data['Date'].dt.year
dt_col_param = []

for col in data.columns:
    dt_col_param.append({"name": str(col), "id": str(col)})



def render_state_message(msg: any) -> html.Div:
    content = []
    content.append(html.Div(msg))
    return html.Div(children=content)


def parse_triggers(triggered) -> list:
    triggers = []
    for t in triggered:
        triggers.append(t["prop_id"])
    return triggers


app.layout = html.Div(children=[

    dcc.Tabs(id="app-tabset", value="app-tab-model", children=[
        dcc.Tab(label="Data", value="app-tab-model", children=[
            html.Button('Оновити дані', id='btn-update', n_clicks=0,
                        style={'font-size': 'large', 'height': 50, 'width': 400, 'background-color': '#e6ecf5',
                               'border-width': 2, 'border-radius': 5, 'border-color': 'lightblue'}),

            html.Br(), html.Br(), html.Br(),
            dash_table.DataTable(
                columns=dt_col_param,
                data=data.to_dict('records'), page_size=10,
                style_header={
                    'backgroundColor': '#e6ecf5',
                },
                id="tbl"

            ),

            # Dash Graph Component calls the prices_graph parameters
            dcc.Graph(figure=plot_btc_prices()), dcc.Graph(figure=plot_oil_prices()),
            dcc.Graph(figure=plot_price_correlation()),
        ]),
        dcc.Tab(label="Model", value="app-tab-data", children=[
            dcc.Graph(id="pred-graph"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=pd.to_datetime(data['Date'].values[-1]),
                max_date_allowed=datetime.date(2024, 1, 1),
                initial_visible_month=datetime.date(2023, 6, 1),
                end_date=datetime.date(2023, 6, 2)
            ),
            html.Div(id='output-container-date-picker-range'),
            html.Button('Зробити прогноз на майбутнє', id='btn-predict', n_clicks=0,
                        style={'font-size': 'large', 'height': 50, 'width': 300, 'background-color': '#e6ecf5',
                               'border-width': 2, 'border-radius': 5, 'border-color': 'lightblue'}),
            dcc.Download(id="predict-download-file"),
            html.Br(), html.Br(),
            html.Button('Навчити модель на нових даних', id='btn-train', n_clicks=0,
                        style={'font-size': 'large', 'height': 50, 'width': 400, 'background-color': '#e6ecf5',
                               'border-width': 2, 'border-radius': 5, 'border-color': 'lightblue'}),
            html.P(render_state_message("No new messages"), id="app-state-message"),
        ]),

    ])
])



@app.callback(Output("btc-graph", "figure"), Output("oil-graph", "figure"),
              Output("corr-graph", "figure"), Output("pred-graph", "figure"),
              Output("predict-download-file", "data"), Output("app-state-message", "children"),
              Input("btn-update", "n_clicks"),
              Input("btn-predict", "n_clicks"),
              Input("btn-train", "n_clicks"),
              State("date-picker-range", "start_date"),
              State("date-picker-range", "end_date"),
              )
def do_all(update_nclicks, predict_nclicks, train_nclicks, pred_start, pred_end):
    output_list = ["btc_graph_figure", "oil_graph_figure", "corr_graph_figure", "pred_graph_figure",
                   "predict_download_file_data", "app_state_message_children"]

    def wrap_output(**kwargs):
        args = kwargs
        res = []
        for o in output_list:
            if o in kwargs.keys():
                res.append(kwargs[o])
            else:
                res.append(dash.no_update)
        return res

    triggers = parse_triggers(dash.callback_context.triggered)

    model = load_saved_model()
    data = load_data()
    bitcoin_prices = data['Close_x'].values.reshape(-1, 1)
    dates = data['Date'].values.reshape(-1, 1)
    oil_prices = data['Close_y'].values.reshape(-1, 1)

    # Нормалізація даних
    scaled_bitcoin_prices, bitcoin_scaler = normalize_data(bitcoin_prices)
    scaled_oil_prices, oil_scaler = normalize_data(oil_prices)

    # Розмір тренувальної вибірки
    train_size = int(len(scaled_bitcoin_prices) * 0.8)

    # Розділення даних на тренувальну та тестову вибірки
    train_bitcoin_data = scaled_bitcoin_prices[:train_size]
    train_oil_data = scaled_oil_prices[:train_size]
    test_bitcoin_data = scaled_bitcoin_prices[train_size:]
    test_oil_data = scaled_oil_prices[train_size:]

    # Розмір вікна
    window_size = 50

    if "btn-train.n_clicks" in triggers:
        X_train, y_train = create_features(train_bitcoin_data, train_oil_data, window_size)
        model = train_model(X_train, y_train, window_size)
        save_model(model)  # Збереження моделі
        return wrap_output(
            app_state_message_children=render_state_message("Training succeeded"))

    if "btn-predict.n_clicks" in triggers:
        days = 7
        X_test, y_test = create_features(test_bitcoin_data, test_oil_data, window_size)
        predictions, future_predictions = make_predictions(model, X_test, bitcoin_scaler, window_size, days)
        test_dates = pd.date_range(end=data['Date'].values[-1], periods=len(X_test) + 1)[1:]
        test_prices = data['Close_x'].values[-len(test_dates):]
        future_dates = pd.date_range(start=data['Date'].values[-1], periods=days + 1)[1:]
        results = pd.DataFrame({'Date': future_dates, 'Price': future_predictions})
        pred_graph = plot_predictions(test_prices, future_predictions, test_dates, future_dates)
        # results.to_csv('C:/Users/stopd/PycharmProjects/scenario_modeling/prediction.csv')
        return wrap_output(graph_figure=pred_graph,
                           app_predict_download_file_data=send_data_frame(results.to_csv, "prediction.csv"),
                           app_state_message_children=render_state_message("Prediction succeeded"))
    return wrap_output()


if __name__ == "__main__":
    app.run_server()
