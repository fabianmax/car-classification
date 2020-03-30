import dash
import dash_bootstrap_components as dbc

from source.data import GameData

# Initialize Dash App with Bootstrap CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Underlying Flask App for productive deployment
server = app.server

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

# Load Game Data
game_data = GameData()
