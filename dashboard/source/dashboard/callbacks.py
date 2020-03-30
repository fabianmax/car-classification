from typing import Dict, List, Tuple

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from app import app, game_data
from source.data import LABELS, ItemLabel


@app.callback(Output('url', 'pathname'), [Input('btn', 'n_clicks')],
              [State('car-brand', 'value'),
               State('car-model', 'value')])
def btn_click_event(n_clicks: int, car_brand: str, car_model: str) -> str:
    """Event callback for all buttons which redirect eventually to a new page

    Arguments:
        n_clicks {int} -- number of clicks
        car_brand {str} -- user predicted car brand
        car_model {str} -- user predicted car model

    Raises:
        PreventUpdate: Sometimes there is a phantom click on page load; Ignore it

    Returns:
        str -- new path
    """
    if n_clicks > 0:
        # Logic for button on result page
        # Due to dash multi page limitations it is not possible to separate the button callbacks
        # Do NOT try to refactor this function:
        # See: https://community.plot.ly/t/you-have-already-assigned-a-callback-to-the-output/25334
        if car_brand == 'ignore' and car_model == 'ignore':
            if game_data.current_round < game_data.max_rounds:
                game_data.current_round += 1
                return '/attempt'

            else:
                return '/finish'

        # Logic for button on attempt page
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
    """Displays an error message if one of the predictions dropwdowns is empty

    Arguments:
        n_clicks {int} -- number of clicks
        car_brand {str} -- user predicted car brand
        car_model {str} -- user predicted car model

    Raises:
        PreventUpdate: Sometimes there is a phantom click on page load; Ignore it

    Returns:
        bool -- False if error, True otherwise
    """
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
def set_model_dropdown(car_brand: str) -> Tuple[bool, List[Dict[str, str]]]:
    """Fill the values for the car model dropdown based on the
    user selection of the car brand dropdown

    Arguments:
        car_brand {str} -- user predicted car brand

    Raises:
        PreventUpdate: If car brand is not fill, ignore everything

    Returns:
        Tuple[bool, List[Dict[str, str]]] -- Dict with car models
    """
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


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n_clicks: int, is_open: bool) -> bool:
    """Toogle navigation bar on mobile devices

    Arguments:
        n_clicks {int} -- number of clicks
        is_open {bool} -- is navigation bar open

    Returns:
        bool -- new state of navigation bar
    """
    if n_clicks:
        return not is_open
    return is_open
