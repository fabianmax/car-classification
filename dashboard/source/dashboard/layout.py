from math import ceil
from typing import Any, Dict, Tuple

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from source.data import LABELS, GameData

COLOR_STATWORX = "#013848"


def count_score(data: GameData) -> Tuple[int, int]:
    """Calculates the current score of user and ai

    Arguments:
        data {GameData} -- Game data

    Returns:
        Tuple[int, int] -- user and ai score
    """
    score_user = score_ai = 0

    for item in data.items:
        if not hasattr(item, "prediction_user"):
            return score_user, score_ai

        if item.prediction_user == item.ground_truth:
            score_user += 1

        if item.prediction_ai[0] == item.ground_truth:
            score_ai += 1

    return score_user, score_ai


def main_layout(app: dash.Dash, data: GameData, content: html) -> html:
    """Main layout which consists of the header, footer
    and a dynamically changing main layout

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data
        content {html} -- layout for the main content

    Returns:
        html -- html layout
    """
    layout = html.Div([
        html.Header(get_header(app, data)),
        html.Main(id='page-content', children=[content]),
        html.Footer(get_footer())
    ])

    return layout


def get_header(app: dash.Dash, data: GameData) -> html:
    """Layout for the header

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data

    Returns:
        html -- html layout
    """
    logo = app.get_asset_url("logo.png")

    score_user, score_ai = count_score(data)

    header = dbc.Container(
        dbc.Navbar(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=logo, height="40px")),
                            dbc.Col(
                                dbc.NavbarBrand("Beat the AI - Car Edition",
                                                className="ml-2")),
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="/",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(dbc.Row(
                    dbc.Col([
                        dbc.Button([
                            "You:",
                            dbc.Badge(score_user, color="light", className="ml-1")
                        ],
                                   className="mr-2"),
                        dbc.Button([
                            "AI:",
                            dbc.Badge(score_ai, color="light", className="ml-1")
                        ])
                    ]),
                    no_gutters=True,
                    className="ml-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                ),
                             id="navbar-collapse",
                             navbar=True),
            ],
            color=COLOR_STATWORX,
            dark=True,
        ),
        className='mb-4 mt-4 navbar-custom')

    return header


def get_footer() -> html:
    """Layout for the footer

    Returns:
        html -- html layout
    """
    footer = dbc.Container([
        html.Hr(),
        html.Div([
            'Made with ❤ in Frankfurt from ',
            dcc.Link(children='STATWORX',
                     href='https://www.statworx.com/',
                     style={"color": COLOR_STATWORX})
        ])
    ],
                           className='mb-4')

    return footer


def start_page(app: dash.Dash, data: GameData) -> html:
    """Layout for the start/index page

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data

    Returns:
        html -- html layout
    """

    data.reset()

    start_page = dbc.Container(
        dbc.Jumbotron([
            html.H1("Can you beat our AI?", className="display-3"),
            html.P(
                "Are you a car expert? You think you know cars better than our machine "
                "learning algorithm? Give it a try!",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.
            P("We'll show you a picture of a car (part) and you'll have to guess "
              "its brand and model. Our AI will do the same. Let's see who's better at "
              "guessing cars in the next 5 rounds! ;)"),
            html.P(dbc.Button("Let's Play!",
                              color="primary",
                              href="/attempt",
                              style={
                                  "background-color": COLOR_STATWORX,
                                  "border-color": COLOR_STATWORX
                              }),
                   className="lead"),
        ]))

    return start_page


def finish_page(app: dash.Dash, data: GameData) -> html:
    """Layout for the finish page. This page get display once all
    rounds are played and the final score is determined.

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data

    Returns:
        html -- html layout
    """
    score_user, score_ai = count_score(data)

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    'You',
                    className="align-items-center d-flex justify-content-center"),
                dbc.CardBody(
                    html.H1(score_user),
                    className="align-items-center d-flex justify-content-center")
            ]),
                    width=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    'AI', className="align-items-center d-flex justify-content-center"),
                dbc.CardBody(
                    html.H1(score_ai),
                    className="align-items-center d-flex justify-content-center")
            ]),
                    width=4)
        ],
                justify="center",
                className="mb-4"),
        dbc.Row(dbc.Col(dbc.Button('Play again!',
                                   id="btn-reset",
                                   block=True,
                                   href="/",
                                   color="primary",
                                   style={
                                       "background-color": COLOR_STATWORX,
                                       "border-color": COLOR_STATWORX
                                   }),
                        width=4),
                justify="center")
    ],
                           className="mb-4")

    return layout


def attempt(app: dash.Dash, data: GameData) -> html:
    """Layout for the attempt page.
    At tis page, the user is able to make his prediction.

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data

    Returns:
        html -- html layout
    """
    idx = data.current_round
    img_raw = str(data.items[idx].picture_raw)

    print('Ground Truth:', data.items[idx].ground_truth)

    layout = dbc.Container([
        html.Div(dbc.Row(dbc.Col(
            dbc.Alert("Please guess the brand and the model!", color="danger")),
                         className='mb-4'),
                 id='error-alert',
                 hidden=True),
        dbc.Row(children=[
            dbc.Col(dbc.Card(dbc.CardImg(src=img_raw))),
            dbc.Col(
                dbc.Card([
                    dbc.FormGroup([
                        dbc.Label("Car Brand", html_for="car-brand"),
                        dcc.Dropdown(
                            id="car-brand",
                            options=[{
                                "label": col,
                                "value": col
                            } for col in list(LABELS.keys())],
                        ),
                        dbc.FormFeedback("Please guess the car brand and car model.",
                                         valid=False)
                    ]),
                    dbc.FormGroup([
                        dbc.Label("Car Model", html_for="car-model"),
                        dcc.Dropdown(id="car-model", disabled=True),
                        dbc.FormFeedback("Please guess the car brand and car model.",
                                         valid=False)
                    ]),
                    dbc.ButtonGroup(
                        dbc.Button("Make a Guess",
                                   id='btn',
                                   color='primary',
                                   style={
                                       "background-color": COLOR_STATWORX,
                                       "border-color": COLOR_STATWORX
                                   }))
                ],
                         body=True))
        ],
                className='mb-4')
    ])

    return layout


def result(app: dash.Dash, data: GameData) -> html:
    """Layout for the result page. This page get display after every user prediction.
    This pages displays the predictions from the user and the ai together with
    the ground truth. An explanation for the ai prediction is also provided.

    Arguments:
        app {dash.Dash} -- dash app instance
        data {GameData} -- game data

    Returns:
        html -- html layout
    """
    item = data.items[data.current_round]
    img_raw = str(item.picture_raw)
    img_explained = item.picture_explained
    ai_prediction: Dict[str, Any] = {'x': [], 'y': []}
    ai_prediction['type'] = 'bar'
    ai_prediction['orientation'] = 'h'
    ai_prediction['marker'] = {'color': COLOR_STATWORX}
    max_axis_value = ceil(item.prediction_ai[0].certainty * 10.0) / 10.0

    # Prepare data for the plot
    for ai_item in item.prediction_ai:
        ai_prediction['y'].append(ai_item.brand + ' ' + ai_item.model)
        ai_prediction['x'].append(ai_item.certainty)

    # Determine result and color
    if item.prediction_user == item.ground_truth:
        clr_user = 'success'

    else:
        clr_user = 'danger'

    if item.prediction_ai[0] == item.ground_truth:
        clr_ai = 'success'

    else:
        clr_ai = 'danger'

    layout = dbc.Container([
        dbc.Row(children=[
            dbc.Col(dbc.Card(dbc.CardImg(src=img_raw))),
            dbc.Col(
                dbc.Card([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            dbc.ListGroupItemHeading("Your Prediction:"),
                            dbc.ListGroupItemText(item.prediction_user.brand + ' - ' +
                                                  item.prediction_user.model)
                        ],
                                          color=clr_user),
                        dbc.ListGroupItem([
                            dbc.ListGroupItemHeading("AI Prediction:"),
                            dbc.ListGroupItemText(item.prediction_ai[0].brand + ' - ' +
                                                  item.prediction_ai[0].model)
                        ],
                                          color=clr_ai),
                        dbc.ListGroupItem([
                            dbc.ListGroupItemHeading("Correct Answer:"),
                            dbc.ListGroupItemText(item.ground_truth.brand + ' - ' +
                                                  item.ground_truth.model)
                        ],
                                          color='secondary',
                                          className='mb-3')
                    ]),
                    dbc.ButtonGroup(
                        dbc.Button("Continue!",
                                   id="btn",
                                   color='primary',
                                   style={
                                       "background-color": COLOR_STATWORX,
                                       "border-color": COLOR_STATWORX
                                   }))
                ],
                         body=True))
        ],
                className='mb-4'),
        dbc.Row(children=[
            dbc.Col(
                dbc.Card([
                    # dbc.CardBody(
                    #    html.H4("How the AI sees the car", className="card-title")),
                    dbc.CardImg(src=img_explained)
                ])),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        # html.H4("Top 5 AI Predictions", className="card-title"),
                        dcc.Graph(figure={
                            'data': [ai_prediction],
                            'layout': {
                                'margin': {
                                    'l': 100,
                                    'r': 0,
                                    'b': 0,
                                    't': 0
                                },
                                'yaxis': {
                                    'automargin': True,
                                    'autorange': 'reversed'
                                },
                                'xaxis': {
                                    'automargin': True,
                                    'tickformat': '.2%',
                                    'range': [0, max_axis_value]
                                },
                                'autosize': True
                            }
                        },
                                  config={
                                      'showTips': False,
                                      'displayModeBar': False,
                                      'doubleClick': False,
                                  },
                                  style={
                                      'flex': 1,
                                      'margin': '10px'
                                  })
                    ]), ))
        ],
                className='mb-4'),
        # Needed to circumvent dash limitations
        # See: https://community.plot.ly/t/you-have-already-assigned-a-callback-to-the-output/25334
        html.Div([
            dcc.Input(id='car-brand', value='ignore'),
            dcc.Input(id='car-model', value='ignore')
        ],
                 hidden=True)
    ])

    return layout
