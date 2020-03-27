
import dash_html_components as html
from dash.dependencies import Input, Output

from launch_dashboard import app
from source.dashboard.layout import attempt, result

# @app.callback(input=Input('input', 'name'))
# def save_name(name: str) -> None:
#     print(name)

@app.callback(Output('content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname: str) -> html:
    if pathname == '/attempt':
        return attempt
    elif pathname == '/result':
        return result
    else:
        pass
