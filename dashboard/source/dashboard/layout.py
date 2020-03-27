from collections import defaultdict
from typing import Tuple

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from source.data import LABELS, GameData

COLOR_STATWORX = "#013848"


def count_score(data: GameData) -> Tuple[int, int]:
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
    layout = html.Div([
        html.Header(get_header(app, data)),
        html.Main(id='page-content', children=[content]),
        html.Footer(get_footer())
    ])

    return layout


def get_header(app: dash.Dash, data: GameData) -> html:
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
            #color="primary",
            dark=True,
        ),
        className='mb-4 mt-4 navbar-custom')

    return header


def get_footer() -> html:
    footer = dbc.Container([
        html.Hr(),
        html.Div([
            'Made with ❤ in Frankfurt from ',
            dcc.Link(children='Statworx',
                     href='https://www.statworx.com/',
                     style={"color": COLOR_STATWORX})
        ])
    ],
                           className='mb-4')

    return footer


def start_page() -> html:
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
            #html.
            #P("Within 5 rounds you can try to beat our AI. We'll show you a picture of a car (part). "
            #  "You have to guess the brand and model of the car. "
            #  "Our AI will do the same. Let's see who's better with cars ;) "),
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
    idx = data.current_round
    #img_raw = app.get_asset_url(str(data.items[idx].picture_raw))
    img_raw = str(data.items[idx].picture_raw)

    print(data.items[idx].ground_truth)

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
    item = data.items[data.current_round]
    #img_raw = app.get_asset_url(str(item.picture_raw))
    #img_explained = app.get_asset_url(str(item.picture_explained))
    img_raw = str(item.picture_raw)
    img_explained = str(item.picture_explained)
    ai_prediction = defaultdict(list)
    ai_prediction['type'] = 'bar'
    ai_prediction['orientation'] = 'h'
    ai_prediction['marker'] = dict(color=COLOR_STATWORX)
    #ai_prediction['layout'] = dict(margin={"l": "2000px"})

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
        dbc.Row(
            children=[
                #dbc.Col(dbc.Card(dbc.CardImg(src=img_explained))),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                figure={'data': [ai_prediction]},
                                config={
                                    'showTips': False,
                                    'displayModeBar': False,
                                    'doubleClick': False
                                },
                                #style={'width': '100%'}
                            )),
                        #className='text-center'
                    ))
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
