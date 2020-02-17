<p float="left">
 <img src="https://github.com/RaffaeleAns/Forecasting-/blob/master/images/DS%20Logo.png" width = "500"/>
 <img src="https://github.com/RaffaeleAns/Forecasting-/blob/master/images/Bicocca%20Logo.png" width = "100" align="right"/>
</p>

# Forecasting
The aim of this project is the definition, development and validation of different forecasting models in order to predict the daily price of Electricity in the Italian Market.
The goal is the daily prediction over a 11 months horizon using 9-years price data with ARIMA, UCM and Machine Leaning models.

<p align = "center">
  <img src="hhttps://github.com/RaffaeleAns/Forecasting-/blob/master/images/forecast.png" width = "500">
</p>  

## 1. The task
The time series consists of daily data from 01/01/2010 to 12/31/2018 (3287 observation) without NULL values. The values range in a wide interval, so a logarithmic transformation is applied to the time series.
The predictions are requested for the interval 01/01/2019 - 11/31/2019 (1 to 334 step ahead). Test set data are unknown, so a partition of the training set has been used for the validation. In order to validate the algorithms on the same horizon of the test set, the validation set has been identified as the last 334 observation of the training set (almost 90% - 10% split), consisting on the interval 02/01/2018 - 12/31/2018. The evaluations and comparison of the different forecasts are in terms of Mean Absolute Error (MAE), that gives a direct economical interpretation of the forecast loss (when applied to the original scale), also for non-statistician experts (usually the requester of the forecast). It is also one of the few metrics available for the LSTM RNN.

## 2. ARIMA
ARMA (Autoregressive Moving Average) models are the most general class of linear models for forecasting a time series through the Box & Jenkins approach (1970). The main idea of this approach is that the model is adjusted towards the reality, in opposition to the classic approach where the reality was supposed to go towards the model. The first operations to identify the model are conducted in order to make the time series stationary. The non stationarity in variance has been corrected with the logarithmic transformation, while for the non stationarity in mean, a seasonal difference is applied. The resulting time series is smoother, but not as much as desired. However, the Augmented Dicky Fuller test confirms the stationarity. The ACF never decays and the PACF has an alternate course and decays after almost 50 lags but reappears at seasonal lags. This suggests the presence of both AR and MA components, maybe a ARIMA(2,1,1)(1,1,1). Unfortunately, the R package TSA doesn’t deal with lags greater than 365, so a seasonal model for daily data can’t be developed. The _auto.arima_ function has been used to identify a model that could be fitted. It select the best models with respect to the AIC score and it gave back a ARIMA(5,1,3) model. The ARIMA forecast on the validation set has good performances for the firsts delays, but it becomes a straight line after few steps ahead, losing all information about sea- sonality, cycle, etc. Its Validation MAE is equal to 0.236.

## 3. UCM
The Unobserved Components Models (known also as Structural Time Series Models) are models where the time se- ries is seen as sum of some unobserved components, typically trend , cycle, seasonality and white noise. Although it seems like the classical approach, the UCM components are defined as stochastic processes, allowing statistical testing, the use of regressors and the derivation of the whole distribution of future outcomes. The Unobserved Components Models have been developed with KFAS package.
External regressors, representing different holidays like Christmas or Easter, have been included in the models.
The identification of the best UCM has been conducted iteratively, from the simplest model with Local Linear Trend and daily stochastic dummy seasonality of order 7, to more complex models with economic cycle and intra-annual daily stochastic trigonometric seasonality. nitial diffuse conditions have been avoided for computational reasons. The mean of the time series has been used as initial condition for the level, while the variance of the components has been initialized with the variance of the process.
The estimation has been made with the Kalman Filter with state smoothing and signal filter (that includes the one-step ahead predictions).
The 8th model, with Local linear Trend, Cycle, Stochastic Seasonality (dummy) of order 7 and Stochastic Seasonality (trigonometric) of order 365 (where only the firsts 16 harmonics have been included) has the best performances on the validation set, with a MAE of only 0.158. This model will be deployed for the test set prediction (after a new training where the training set includes also the validation set).

## 4. Machine Learning Models
Two Machine Learning models have been developed, the KNN regression and a LSTM (Long Short-Term Recurrent) Neural Network.
Compared with classical statistical models, computational intelligence methods exhibit interesting features, such as non- linearity or the lack of an underlying model, that is, they are non-parametric.

- KNN is a simple algorithm that stores a collection of examples (consisting of a vector the historical values of the time series and his associated target values) and, given a new example, KNN finds its k most similar examples (called nearest neighbors) according to the Euclidean distance. Then, the prediction is performed as an aggregation of the target values associated with its nearest neighbors.
Two strategies have been utilized to perform the prediction: The MIMO (multiple Input Multiple Output) strategy, that is characterized by the use of a vector of target values, and the recursive strategy, an iterative one-step ahead strategy, similar forecasting method of ARIMA and Kalman filter. The KNN models have been developed with tsfknn package.
The first model uses the MIMO strategy and it is extremely fast to train and make the predictions. It uses 1:1500 lags and 3 nearest neighbors to perform the prediction.
The second model uses the recursive strategy, this means that only one-step ahead forecasts are performed iteratively to forecast all the 334 future observations (the model uses previous predictions instead of historical values when these are unavailable). The parameters lags and number of nearest neighbors are the same as the previous model.
The results are pretty similar between the two strategies.

- The LSTM is a Recurrent Neural Network (so with shared parameters) that reduces the effect of vanishing gradients during the backpropagation with connection weights that may change at every time step. It is composed of cells which include several operations. An internal state variable is passed and modified into the cells through the Operation Gates (Forget Gate, Input Gate, Output Gate). The Long Short Term Memory Recurrent Neural Network has been developed with the help of Keras package.
A generator function has been used to transform the data in a suitable form for the LSTM. In particular, every batch contains 128 records with a lookback (the past values taken in account) equals to 1500 and 334 future target values. The validation data has been defined with the same generator function with some adjustments in order to obtain the same validation window as the previous models.
The model architecture consists of one lstm layer with 32 units, tanh activation function 0.1 dropout rate and 0.3 recurrent dropout rate and one dense output layer with 334 neurons (equals to target horizon).
Once compiled with rmsprop optimizer and mae loss func- tion, the model is trained over 10 epoch with 50 steps per epoch.
The validation MAE (≈ 0.26) is similar to the training MAE (≈ 0.20).

## 5. Conclusions

The aim of this paper was to explore the peculiarities of each of the three forecasting methods developed. Starting from the ARIMA forecasting, it has not proved entirely satisfactory, although its error measures are not so bad. The difficulties to deal with daily data in R contributed to the lack of specialization on the models. ARIMA models seemed to be more 
suitable for an high level time aggregation (e.g. week or month) and more regular time series.
The UCM models have revealed very tricky in the construction. The freedom of KFAS package allow to develop ad-hoc models with particular update functions and parameters to be estimated. Also, the easily addiction of external features is a great feature of UCM.
The two Machine Learning models revealed different properties. The KNN regressions are extremely fast to train and predict, and they need almost 4 or 5 rows of code to be implemented, that makes these models extremely easy, without a great loss of performances. The use of other distance metrics, that weights more the latest observation than the remote ones, can reveal interesting results.
The LSTM RNNs are deeper models, that can be fine tuned with respect of many hyper-parameters, with the drawback of an extremely long time of training (almost 1 hour in this case). The feeling is that with more computational power, more data and the right hyper-parameters tuning, the RNN can outperforms the other forecasting methods.
Finally, the analysis revealed that ARIMA, UCM and machine learning models are able ( with some differences on the performances) to predict the future values of the Italian Market electricity price.



<p align = "center">
  <img src="https://github.com/RaffaeleAns/Forecasting-/blob/master/images/AR%20Logo.png" width = "250">
</p>    
    
    
<p align = "center">
<a href="https://github.com/RaffaeleAns">
<img border="0" alt="W3Schools" src="https://github.com/RaffaeleAns/Forecasting-/blob/master/images/GitHub%20Logo.png" width="20" height="20">
</a>
 <a href="https://www.linkedin.com/in/raffaele-anselmo-213a0a179">
<img border="0" alt="do" src="https://github.com/RaffaeleAns/Forecasting-/blob/master/images/LinkedIn%20Logo.png" width="20" height="20">
</a>
</p>


