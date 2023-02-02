from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import os

ek.set_app_key(os.getenv('EIKON_API'))

# dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H4('benchmark',style={'display':'inline-block','margin-right':10}),
        dcc.Input(id = 'benchmark-id', type = 'text', value="IVV", style={"margin-right": "25px"}),
        html.H4('asset',style={'display':'inline-block','margin-right':10}),
        dcc.Input(id = 'asset-id', type = 'text', value="AAPL.O", style={"margin-right": "25px"}),
        html.H4('date range',style={'display':'inline-block','margin-right':10}),
        dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(1995, 8, 5),
                # max_date_allowed=date(2017, 9, 19),
                max_date_allowed=date(datetime.now().year,
                                      datetime.now().month,
                                      datetime.now().day),
                initial_visible_month=date(2017, 8, 5),
                start_date=date(2017, 5, 20),
                end_date=date(2017, 8, 25),
                style={"margin-right": "25px"}
            ),
            html.Div(id='output-container-date-picker-range')
    ]),
    html.Button('QUERY Refinitiv', id = 'run-query', n_clicks = 0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot'),
    html.H4('plot date range',style={'display':'inline-block','margin-right':10}),
    dcc.DatePickerRange(
                    id='plot-date-picker-range',
                    min_date_allowed=date(1995, 8, 5),
                    # max_date_allowed=date(2017, 9, 19),
                    max_date_allowed=date(datetime.now().year,
                                          datetime.now().month,
                                          datetime.now().day),
                    initial_visible_month=date(2017, 8, 5),
                    start_date=date(2017, 5, 20),
                    end_date=date(2017, 8, 25)
                ),
                html.Div(id='plot-container-date-picker-range'),
    html.Button('Update plot', id = 'run-plot', n_clicks = 0),
    html.P(id='result-text', children=""),
    dcc.Graph(id="ab-plot"),
    html.P(id='summary-text', children=""),
    dcc.Store(id='full-history-tbl')
])

# when user update the date range in refinitiv data fetch, automatically update
# the available date range of plot graph data, can avoid user's wrong operation
@app.callback(
    Output('plot-date-picker-range', 'start_date'),
    Output('plot-date-picker-range', 'end_date'),
    Output('plot-date-picker-range', 'min_date_allowed'),
    Output('plot-date-picker-range', 'max_date_allowed'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    # prevent_initial_call=True
)
def date_ranging(start_date, end_date):
    return start_date, end_date, start_date, end_date

# fetch raw data from refinitiv
@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),
     State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, startdate, enddate):
    assets = [benchmark_id, asset_id]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            # 'SDate': '2018-01-01',
            # 'EDate': datetime.now().strftime("%Y-%m-%d"),
            # 'Frq': 'D'
            'SDate': startdate,
            'EDate': enddate,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            # 'SDate': '2018-01-01',
            # 'EDate': datetime.now().strftime("%Y-%m-%d"),
            # 'Frq': 'D'
            'SDate': startdate,
            'EDate': enddate,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            # 'SDate': '2018-01-01',
            # 'EDate': datetime.now().strftime("%Y-%m-%d"),
            # 'Frq': 'D'
            'SDate': startdate,
            'EDate': enddate,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return(unadjusted_price_history.to_dict('records'))

# input raw history table and calculate the returns table
@app.callback(
    Output("returns-tbl", "data"),
    Output("full-history-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    fulltbl = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    })
    return(
        #     pd.DataFrame({
        #     'Date': numerator[dte_col].reset_index(drop=True),
        #     'Instrument': numerator[ins_col].reset_index(drop=True),
        #     'rtn': np.log(
        #         (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
        #                 denominator[prc_col] * denominator[spt_col]
        #         ).reset_index(drop=True)
        #     )
        # }).pivot_table(
        #     values='rtn', index='Date', columns='Instrument'
        # ).to_dict('records')

        fulltbl.pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).to_dict('records'), fulltbl.to_dict('records')
        # pd.DataFrame(fulltbl.to_dict('records')).pivot_table(
        #     values='rtn', index='Date', columns='Instrument'
        # ).to_dict('records'), fulltbl.to_dict('records')


    )

# get the full history table and sort the data by input date
# then plot the graph by filtered data and retrieve and update the Alpha and Beta
@app.callback(
    Output("ab-plot", "figure"),
    Output("result-text", "children"),
    Input("run-plot", "n_clicks"),
    Input("returns-tbl", "data"),
    Input("full-history-tbl", "data"),
    # add the date range as input, every time update,
    # just trigger the callback to update the graph

    # [State('benchmark-id', 'value'), State('asset-id', 'value'),
    #  State('plot-date-picker-range')],
    [State('benchmark-id', 'value'), State('asset-id', 'value'), State('plot-date-picker-range', 'start_date'),
    State('plot-date-picker-range', 'end_date')],
    # Input('plot-date-picker-range', 'start_date'),
    # Input('plot-date-picker-range', 'end_date'),
    prevent_initial_call = True
)
def render_ab_plot(n_clicks, returns, date_returns, benchmark_id, asset_id, startdate, enddate):
    data = pd.DataFrame(date_returns)

    range_rtns = data[(data['Date'] >= (startdate+'T00:00:00'))
                      &(data['Date'] <= (enddate+'T00:00:00'))]

    filt_range_rtns = range_rtns.pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).to_dict('records')

    resgraph = px.scatter(filt_range_rtns, x=benchmark_id, y=asset_id, trendline='ols')
    # results = px.get_trendline_results(resgraph).px_fit_results.iloc[0].summary()
    model = px.get_trendline_results(resgraph)
    results = model.iloc[0]["px_fit_results"]
    # alpha = results.params[0]
    # beta =results.params[1]
    # p_beta =.pvalues[1]
    # r_squared =.rsquared
    alpha = '{:.6f}'.format(results.params[0])
    beta = '{:.6f}'.format(results.params[1])
    return(
        resgraph, "Alpha:"+str(alpha)+", Beta:"+str(beta)
    )

if __name__ == '__main__':
    app.run_server(debug=True)