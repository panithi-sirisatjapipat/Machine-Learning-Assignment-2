import dash
import mlflow 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import warnings


class LinearRegression(object):
    
    
    kfold = KFold(n_splits=3)
     # Constructor for the LinearRegression class        
    def __init__(self, regularization, lr, theta_init, momentum, method, num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr # Learning rate
        self.num_epochs = num_epochs # Number of training epochs
        self.batch_size = batch_size # Batch size for mini-batch training
        self.method     = method # Training method ('sto' for stochastic, 'mini' for mini-batch, 'batch' for batch)
        self.theta_init = theta_init # Weight initialization method ('zeros' or 'xavier')
        self.momentum = momentum # Momentum term for gradient descent
        self.cv         = cv # Cross-validation strategy (KFold in this case)
        self.regularization = regularization # Regularization method (L1, L2, etc.)

    # Define Mean Squared Error (MSE) function
    def mse(self, ytrue, ypred):
        return ((ytrue - ypred) ** 2).sum() / ypred.shape[0]
    # Define R-squared (R2) function
    def r2(self, ytrue, ypred):
        return 1 - ((((ytrue - ypred) ** 2).sum()) / (((ytrue - ytrue.mean()) ** 2).sum()))
    
    def fit(self, X_train, y_train):
        # Fit the linear regression model    
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty
        # mlflow.log_params(params=params)  # THEL LINE U CHANGED 

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if self.theta_init == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.theta_init == 'xavier':
                m = X_train.shape[0]
                # calculate the range for the weights
                lower , upper = -(1.0 / math.sqrt(m)), (1.0 / math.sqrt(m))
                # summarize the range
                print(lower , upper)
                
                # generate random numbers
                numbers = np.random.rand(X_cross_train.shape[1])
                self.theta = lower + numbers * (upper - lower)
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__} 
                mlflow.log_params(params=params) 
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    # mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    # mlflow.log_input(mlflow_train_data, context="training")
                    
                    # mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    # mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")

    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        prev_step = 0
        
        if self.momentum == "without":
            step = self.lr * grad
        else:
            step = self.lr * grad + self.momentum * prev_step

        self.theta -= step
        prev_step = step
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def plot_feature_importance(self, feature_names):
        import matplotlib.pyplot as plt
        feature_importance = np.abs(self._coef())
        sorted_indices = np.argsort(feature_importance)
        sorted_features = [feature_names[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, feature_importance[sorted_indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.show()

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): 
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, theta_init, momentum, method)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, theta_init, momentum, method)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, theta_init, momentum, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, theta_init, momentum, method)


app = dash.Dash(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
app.config.suppress_callback_exceptions=True

# Define styles for different elements
headline_style = {"fontSize": 24, "textAlign": "center", "color": "#D1F2EB"}
instruction_style = {"fontSize": 16, "textAlign": "center", "color": "#FEF9E7", "margin": "10px"}
input_style = {"width": "150px", "margin": "10px", "color": "#000000", "backgroundColor": "#FFFFFF"}
submit_button_style = {"textAlign": "center", "marginTop": "20px", "backgroundColor": "#76D7C4", "borderRadius": "20px"}
car_price_style = {"fontSize": 20, "textAlign": "center", "marginTop": "20px", "color": "#FFFFFF"}

# Define the car price prediction page layout
car_price_layout = html.Div(
    style={"backgroundColor": "#154360", "padding": "20px"},
    children=[
        html.H1("Car Price Prediction", style=headline_style),
        html.Div(
            "Fill in the values below to predict the car price:",
            style=instruction_style,
        ),
        html.Div(
            [
                html.Label("Max power (bhp):", style={"color": "#FFFFFF"}),
                dcc.Input(id="max_power", type="number", style=input_style),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Mileage (kmpl):", style={"color": "#FFFFFF"}),
                dcc.Input(id="mileage", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Div(
            [
                html.Label("Engine size (cc):", style={"color": "#FFFFFF"}),
                dcc.Input(id="engine", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Button("Submit", id="submit_car_price", style=submit_button_style, className="rounded-button"),
        html.Div(id="car_price_result", style=car_price_style),
        dcc.Link("Model 2", href="/other_model",
         style={"fontSize": 16, "color": "#FFFFFF", "textDecoration": "underline", "position": "absolute", "top": "0", "right": "10px", "margin": "10px"}),
        dcc.Link("Home", href="/",
         style={"fontSize": 16, "color": "#FFFFFF", "textDecoration": "underline", "position": "absolute", "top": "0", "right": "80px", "margin": "10px"}),

    ],
)

# Define the other model page layout
other_model_layout = html.Div(
    style={"backgroundColor": "#154360", "padding": "20px"},
    children=[
        html.H1("Car Price Prediction Model 2", style=headline_style),
        html.Div(
            "Fill in the values below to predict the car price:",
            style=instruction_style,
        ),
        html.Div(
            [
                html.Label("Max power (bhp):", style={"color": "#FFFFFF"}),
                dcc.Input(id="max_power_other_model", type="number", style=input_style),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Mileage (kmpl):", style={"color": "#FFFFFF"}),
                dcc.Input(id="mileage_other_model", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Div(
            [
                html.Label("Engine size (cc):", style={"color": "#FFFFFF"}),
                dcc.Input(id="engine_other_model", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Button("Submit", id="submit_other_model", style=submit_button_style, className="rounded-button"),
        html.Div(id="other_model_result", style=car_price_style),
        dcc.Link("Model 1", href="/car_price",
         style={"fontSize": 16, "color": "#FFFFFF", "textDecoration": "underline", "position": "absolute", "top": "0", "right": "10px", "margin": "10px"}),
        dcc.Link("Home", href="/",
         style={"fontSize": 16, "color": "#FFFFFF", "textDecoration": "underline", "position": "absolute", "top": "0", "right": "80px", "margin": "10px"}),
        html.Div(
            "Use this new model to make car price predictions based on the provided inputs. Fill in the values for max power, mileage, and engine size, and click 'Submit' to see the predicted car price.",
            style={"fontSize": 16, "color": "#FFFFFF", "margin": "10px"}
        ),

    ],
)

# Load the trained models for each page
car_price_model_data = joblib.load('C:\\Users\\Panithi\\Desktop\\AIT\\DSAI\\ML\\Assignment 2_1\\app\\code\\Car-price.model')
other_model_data = joblib.load('C:\\Users\\Panithi\\Desktop\\AIT\\DSAI\\ML\\Assignment 2_1\\app\\code\\Car-price-2.model')

# Extract the model and scaler from the dictionary
car_price_model = car_price_model_data['model']
car_price_scaler = car_price_model_data['scaler']

other_model = other_model_data['model']
other_scaler = other_model_data['scaler']

# Create a callback function to predict the car price and display it on the car price page
@app.callback(
    Output("car_price_result", "children"),
    [Input("submit_car_price", "n_clicks")],
    [State("max_power", "value"), State("mileage", "value"), State("engine", "value")],
)
def predict_car_price(n_clicks, max_power, mileage, engine):
    if n_clicks is None:
        return ""
    # Perform prediction using the car price model
    input_data = [[max_power, mileage, engine]]
    input_data = car_price_scaler.transform(input_data)
    car_price = car_price_model.predict(input_data)[0]

    return f"The predicted car price is ${car_price:.2f}."

# Create a callback function to predict values for the other model and display them on the other model page
@app.callback(
    Output("other_model_result", "children"),
    [Input("submit_other_model", "n_clicks")],
    [State("max_power_other_model", "value"), State("mileage_other_model", "value"), State("engine_other_model", "value")],
)
def predict_other_model(n_clicks, max_power, mileage, engine):
    if n_clicks is None:
        return ""

    # Perform prediction using the other model
    input_data = [[max_power, mileage, engine]]
    input_data = other_scaler.transform(input_data)
    car_price = other_model.predict(input_data)[0]

    return f"The predicted car price is ${car_price:.2f}."

# Define the callback to switch between pages
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/car_price":
        return car_price_layout
    elif pathname == "/other_model":
        return other_model_layout
    else:
        return homepage_layout  # Use the homepage layout here

# Define the homepage layout
homepage_layout = html.Div(
    style={"backgroundColor": "#154360", "height": "100vh", "display": "flex", "flexDirection": "column",
           "alignItems": "center", "justifyContent": "center"},
    children=[
        html.H1("Welcome to the Models For Car Price Prediction", style=headline_style),
        html.Div(
            "Please kindly choose prefered car price prediction model below",
            style=instruction_style,
        ),
        html.Div([
            dcc.Link(
                html.Button("Car Price Prediction Model 1", style={"fontSize": 15, "color": "#FFFFFF", "backgroundColor": "#3498DB", "borderRadius": "20px"}),
                href="/car_price",
                style={"color": "#FFFFFF", "textDecoration": "none", "margin": "10px"},
            ),
            dcc.Link(
                html.Button("Car Price Prediction Model 2", style={"fontSize": 15, "color": "#FFFFFF", "backgroundColor": "#3498DB", "borderRadius": "20px"}),
                href="/other_model",
                style={"color": "#FFFFFF", "textDecoration": "none", "margin": "10px"},
            ),
        ]),
    ],
)

# Define the layout for the homepage
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
