from typing import Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from source.dashboard import (attempt, finish_page, main_layout, result, start_page)
from source.data import LABELS, GameData, ItemLabel

# Load Game Data
game_data = GameData()

# Initialize Dash App with Bootstrap CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Underlying Flask App for productive deployment
server = app.server

# Set title
app.title = 'Beat the AI - Car Edition'

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.suppress_callback_exceptions = True

# in order to work on shinyproxy
# see https://support.openanalytics.eu/t/what-is-the-best-way-of-delivering-static-assets-to-the-client-for-custom-apps/363/5
app.config.update({
    # as the proxy server will remove the prefix
    'routes_pathname_prefix': '',
    # the front-end will prefix this string to the requests
    # that are made to the proxy server
    'requests_pathname_prefix': ''
})

# Set Layout
app.layout = dbc.Container(
    [dcc.Location(id='url', refresh=False),
     html.Div(id='main-page')])


@app.callback(Output('main-page', 'children'), [Input('url', 'pathname')])
def display_page(pathname: str) -> html:
    if pathname == '/attempt':
        return main_layout(app, game_data, attempt(app, game_data))

    elif pathname == '/result':
        return main_layout(app, game_data, result(app, game_data))

    elif pathname == '/finish':
        return main_layout(app, game_data, finish_page(app, game_data))

    elif pathname == '/':
        return main_layout(app, game_data, start_page())

    else:
        raise PreventUpdate


@app.callback(Output('url', 'pathname'), [Input('btn', 'n_clicks')],
              [State('car-brand', 'value'),
               State('car-model', 'value')])
def btn_click_event(n_clicks: int, car_brand: str, car_model: str) -> str:
    if n_clicks > 0:
        # Logic for result button
        # Due to dash multi page limitations it is not possible to separate the button callbacks
        # Do NOT try to refactor this function:
        # See: https://community.plot.ly/t/you-have-already-assigned-a-callback-to-the-output/25334
        if car_brand == 'ignore' and car_model == 'ignore':
            if game_data.current_round < game_data.max_rounds:
                game_data.current_round += 1
                return '/attempt'

            else:
                return '/finish'

        else:
            if car_brand is not None and car_model is not None:
                prediction_user = ItemLabel(car_brand, car_model)

                idx = game_data.current_round
                game_data.items[idx].prediction_user = prediction_user

                return '/result'

    raise PreventUpdate


@app.callback(Output('error-alert', 'hidden'), [Input('btn', 'n_clicks')],
              [State('car-brand', 'value'),
               State('car-model', 'value')])
def is_dropdown_empty(n_clicks: int, car_brand: str, car_model: str) -> bool:
    if n_clicks is not None and n_clicks > 0:
        if car_brand is None or car_model is None:
            return False

        else:
            return True

    raise PreventUpdate


@app.callback(
    [Output('car-model', 'disabled'),
     Output('car-model', 'options')],
    [Input('car-brand', 'value')],
)
def set_model_dropdown(car_brand: str) -> Tuple[bool, List[Optional[Dict[str, str]]]]:
    if not car_brand == 'ignore':
        if car_brand is not None:
            car_labels = [{
                "label": col,
                "value": col
            } for col in sorted(LABELS[car_brand])]

            return False, car_labels

        else:
            return True, []

    raise PreventUpdate


@app.callback(Output("btn-reset", "hidden"), [Input("btn-reset", "n_clicks")])
def reset(n_clicks: int) -> None:
    if n_clicks > 0:
        global game_data
        game_data = GameData()

    raise PreventUpdate


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n: int, is_open: bool) -> bool:
    if n:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True, port=8050)
