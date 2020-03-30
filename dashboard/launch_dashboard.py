

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app import app, game_data
from source.dashboard import (attempt, callbacks, finish_page,  # noqa: F401
                              main_layout, result, start_page)

# Set title
app.title = 'Beat the AI - Car Edition'

# Set Layout
app.layout = dbc.Container(
    [dcc.Location(id='url', refresh=False),
     html.Div(id='main-page')])


@app.callback(Output('main-page', 'children'), [Input('url', 'pathname')])
def display_page(pathname: str) -> html:
    """Function to define the routing. Mapping routes to layout.

    Arguments:
        pathname {str} -- pathname from url/browser

    Raises:
        PreventUpdate: Unknown/Invalid route, do nothing

    Returns:
        html -- layout
    """
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


if __name__ == "__main__":
    # Execute this file to run the debug server
    # Inside the docker container, a productive server is launched
    app.run_server(host='0.0.0.0', debug=True, port=8050)
