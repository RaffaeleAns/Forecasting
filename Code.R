library(xts)
library(KFAS)

time_series_dataset

y <- time_series_dataset$value
y <- xts(y, as.Date((time_series_dataset$Data),
                    format = "%Y-%m%-d"))
plot(y)

# fix train set and make model
train_length <-round(length(y)*0.8) # 80% Training Set - 20% Validation Set
y[train_length]

y_train <- (as.numeric(y[1:train_length]))

val <- (train_length+1):length(y)
y_val <- y[val]

dum <- read.csv("dummies_updt.csv", sep=";")
dum <- dum[c(1:3287),]
dum <- xts(dum[, -(1:2)], as.Date((time_series_dataset$Data),
                                  format = "%Y-%m%-d"))

Xtrain <- as.data.frame(dum["2010-01-01/2017-03-14"])
Xtrain$y <- y_train


### MODELLO 1 ####
# IRW + CICLO STOCASTICO +  TREND GIORNALIERO SETTIMANALE (DUMMY) STOCASTICO 
# + TREND GIORNALIERO ANNUALE (TRIG) DETERMINISTICO

mod1 <- SSModel(y_train ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(0, NA))  +
                  SSMcycle(1825, NA) +
                  SSMseasonal(7, NA, "dummy") +
                  SSMseasonal(365, 0, "trig", harmonics = 1:20),
                H = NA,
                data = Xtrain)

# We want to avoid diffuse initial conditions on the state variables
# Let us fix their means and variances to reasonable values
vary <- var(y_train)
mod1$a1["level", 1] <- mean(y_train)
mod1$P1inf <- mod1$P1inf * 0
diag(mod1$P1) <- vary


# Initial values for the variances we have to estimate
init <- numeric(6)
init[1] <- log(vary/100)  # log-var(err. slope)
init[2] <- log(vary/10)   # log-var(err. cycle)
init[3] <- log(vary/100)  # log-var(err. dummy seas)
init[4] <- log(vary/100)  # log-var(err.oss)
init[5] <- 3.5            # sigmoid^(-1)(rho) (damping factor)
init[6] <- 0              # sigmoid^(-1)(period, 24, 96)

sigmoid <- function(x, a = 0, b = 1) {
  a + (b - a)/(1 + exp(-x))
}

updt <- function(pars, model) {
  model$Q[2, 2, 1] <- exp(pars[1])                           #slope
  model$Q[3, 3, 1] <- exp(pars[3])                           #seas dummy
  model$Q[44, 44, 1] <- model$Q[45, 45, 1] <- exp(pars[2])   #cycle
  rho <- sigmoid(pars[5]) * 0.99
  per <- sigmoid(pars[6], 365, 1825)
  lam <- 2*pi/per
  vpsi <- model$Q[44, 44, 1] / (1 - rho^2)
  rho_co <- rho*cos(lam)
  rho_si <- rho*sin(lam)
  model$T[44:45, 44:45, 1] <- c(rho_co, -rho_si,
                                rho_si, rho_co)
  model$P1inf[44, 44] <- model$P1inf[45, 45] <- 0
  model$P1[44, 44] <- model$P1[45, 45] <- vpsi
  model$H[1, 1, 1] <- exp(pars[4])
  model
}

# Estimate
fit1 <- fitSSM(mod1, init, updt, control = list(maxit = 2000))


# Smoothing
smo1 <- KFS(fit1$model, smoothing = c("state", "signal"), maxiter=200)


smo1_seas <- rowSums(smo1$alphahat[, seq(23,42,2)])


plot(y_train, type = "l")
lines(smo1$alphahat[, "level"], col = "red")      #Trend

lines(smo1$alphahat[, "level"] +
        smo1$alphahat[, "cycle"], col = "yellow") #Trend + Stagionalità intra-annua

lines(smo1$alphahat[, "level"] +
        smo1_seas, col = "blue")                  #Trend + Stagionalità intra-annua

plot(y_train, type = "l")
lines(smo1$alphahat[, "level"] +
        smo1$alphahat[, "sea_dummy1"] +
        smo1$alphahat[, "cycle"] +                #Trend + Stagionalità intra-annua +
        smo1_seas, col = "green")                 #Stag settimanale + ciclo

# Let us produce one-step-ahead predictions using the model estimated
# on the train set

Xtot <- as.data.frame(dum["2010-01-01/"])
y_tot <- c(y_train, rep(NA, length(y_val)))
Xtot$y <- as.numeric(y_tot)

est <- exp(fit1$optim.out$par)

mod11 <- SSModel(y ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(0, est[1])) +
                  SSMcycle(60, est[2]) +
                  SSMseasonal(7, est[3], "dummy") +
                  SSMseasonal(365, 0, "trig",
                              harmonics = 1:20),
                H = est[4],
                data = Xtot)

# let us fix again the initial conditions so that they are not diffuse
mod11$a1 <- mod1$a1
mod11$P1 <- mod1$P1
mod11$P1inf <- mod1$P1inf

# Let us get smoother and signal filter (it includes one step ahead predictions)
smo11 <- KFS(mod11, smoothing = "state", filter = "signal")
one_step_ahead <- xts(smo11$m[, 1], time(y["2010-01-01/"]))

plot(y["2017-03-14/"])
lines(one_step_ahead["2017-03-14/"], col = "red")

# Compute RMSE
sqrt(mean((one_step_ahead["2017-03-14/"] - y["2017-03-14/"])^2))

# Compute MAPE
mean(abs(one_step_ahead["2017-03-14/"] - y["2017-03-14/"]) / y["2017-03-14/"]) * 100




### MODELLO 2 ####
# IRW + CICLO STOCASTICO +  TREND GIORNALIERO SETTIMANALE (DUMMY) STOCASTICO 
# + TREND GIORNALIERO ANNUALE (TRIG) STOCASTICO

mod2 <- SSModel(y_train ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(0, NA))  +
                  SSMcycle(1825, NA) +
                  SSMseasonal(7, NA, "dummy") +
                  SSMseasonal(365, NA, "trig", harmonics = 1:20),
                H = NA,
                data = Xtrain)

# We want to avoid diffuse initial conditions on the state variables
# Let us fix their means and variances to reasonable values
vary <- var(y_train)
mod2$a1["level", 1] <- mean(y_train)
mod2$P1inf <- mod22$P1inf * 0
diag(mod2$P1) <- vary


# Initial values for the variances we have to estimate
init <- numeric(7)
init[1] <- log(vary/100)  # log-var(err. slope)
init[2] <- log(vary/10)   # log-var(err. cycle)
init[3] <- log(vary/100)  # log-var(err. dummy seas)
init[4] <- log(vary/100)  # log-var(err. trig seas)
init[5] <- log(vary/100)  # log-var(err.oss)
init[6] <- 3.5            # sigmoid^(-1)(rho) (damping factor)
init[7] <- 0              # sigmoid^(-1)(period, 24, 96)

sigmoid <- function(x, a = 0, b = 1) {
  a + (b - a)/(1 + exp(-x))
}

updt2 <- function(pars, model) {
  model$Q[2, 2, 1] <- exp(pars[1])                           #slope
  model$Q[3, 3, 1] <- exp(pars[3])                           #seas dummy
  diag(model$Q[4:43, 4:43, 1]) <- exp(pars[4])               #seas trig
  model$Q[44, 44, 1] <- model$Q[45, 45, 1] <- exp(pars[2])   #cycle
  rho <- sigmoid(pars[6]) * 0.99
  per <- sigmoid(pars[7], 365, 1825)
  lam <- 2*pi/per
  vpsi <- model$Q[44, 44, 1] / (1 - rho^2)
  rho_co <- rho*cos(lam)
  rho_si <- rho*sin(lam)
  model$T[44:45, 44:45, 1] <- c(rho_co, -rho_si,
                                rho_si, rho_co)
  model$P1inf[44, 44] <- model$P1inf[45, 45] <- 0
  model$P1[44, 44] <- model$P1[45, 45] <- vpsi
  model$H[1, 1, 1] <- exp(pars[5])
  model
}

# Estimate
fit2 <- fitSSM(mod2, init, updt2, control = list(maxit = 2000))


# Smoothing
smo2 <- KFS(fit2$model, smoothing = c("state", "signal"), maxiter=200)


smo2_seas <- rowSums(smo2$alphahat[, seq(23,62,2)])
plot(y_train, type = "l")
lines(smo2$alphahat[, "level"], col = "red")      #Trend

lines(smo2$alphahat[, "level"] +
        smo2$alphahat[, "cycle"], col = "yellow") #Trend + Stagionalità intra-annua

lines(smo2$alphahat[, "level"] +
        smo2_seas, col = "blue")                  #Trend + Stagionalità intra-annua

plot(y_train, type = "l")
lines(smo2$alphahat[, "level"] +
        smo2$alphahat[, "sea_dummy1"] +
        smo2$alphahat[, "cycle"] +                #Trend + Stagionalità intra-annua +
        smo2_seas, col = "green")                 #Stag settimanale + ciclo

# Let us produce one-step-ahead predictions using the model estimated
# on the train set

Xtot <- as.data.frame(dum["2010-01-01/"])
y_tot <- c(y_train, rep(NA, length(y_val)))
Xtot$y <- as.numeric(y_tot)

est_2 <- exp(fit2$optim.out$par)

mod22 <- SSModel(y ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(0, est_2[1])) +
                  SSMcycle(60, est_2[2]) +
                  SSMseasonal(7, est_2[3], "dummy") +
                  SSMseasonal(365, est_2[4], "trig",
                              harmonics = 1:20),
                H = est_2[5],
                data = Xtot)

# let us fix again the initial conditions so that they are not diffuse
mod22$a1 <- mod2$a1
mod22$P1 <- mod2$P1
mod22$P1inf <- mod2$P1inf

# Let us get smoother and signal filter (it includes one step ahead predictions)
smo22 <- KFS(mod22, smoothing = "state", filter = "signal")
one_step_ahead_2 <- xts(smo22$m[, 1], time(y["2010-01-01/"]))

plot(y["2017-03-14/"])
lines(one_step_ahead_2["2017-03-14/"], col = "red")

# Compute RMSE
sqrt(mean((one_step_ahead_2["2017-03-14/"] - y["2017-03-14/"])^2))

# Compute MAPE
mean(abs(one_step_ahead_2["2017-03-14/"] - y["2017-03-14/"]) / y["2017-03-14/"]) * 100


### MODELLO 3 ####
# LLT + CICLO STOCASTICO +  TREND GIORNALIERO SETTIMANALE (DUMMY) STOCASTICO 
# + TREND GIORNALIERO ANNUALE (TRIG) STOCASTICO

mod3 <- SSModel(y_train ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(NA, NA))  +
                  SSMcycle(1825, NA) +
                  SSMseasonal(7, NA, "dummy") +
                  SSMseasonal(365, NA, "trig", harmonics = 1:20),
                H = NA,
                data = Xtrain)

# We want to avoid diffuse initial conditions on the state variables
# Let us fix their means and variances to reasonable values
vary <- var(y_train)
mod3$a1["level", 1] <- mean(y_train)
mod3$P1inf <- mod3$P1inf * 0
diag(mod3$P1) <- vary


# Initial values for the variances we have to estimate
init <- numeric(8)
init[1] <- log(vary/10)  # log-var(err. slope)
init[2] <- log(vary/100)  # log-var(err. slope)
init[3] <- log(vary/10)   # log-var(err. cycle)
init[4] <- log(vary/100)  # log-var(err. dummy seas)
init[5] <- log(vary/100)  # log-var(err. trig seas)
init[6] <- log(vary/100)  # log-var(err.oss)
init[7] <- 3.5            # sigmoid^(-1)(rho) (damping factor)
init[8] <- 0              # sigmoid^(-1)(period, 24, 96)

sigmoid <- function(x, a = 0, b = 1) {
  a + (b - a)/(1 + exp(-x))
}

updt3 <- function(pars, model) {
  model$Q[1, 1, 1] <- exp(pars[1])                           #intercept
  model$Q[2, 2, 1] <- exp(pars[2])                           #slope
  model$Q[3, 3, 1] <- exp(pars[4])                           #seas dummy
  diag(model$Q[4:43, 4:43, 1]) <- exp(pars[5])               #seas trig
  model$Q[44, 44, 1] <- model$Q[45, 45, 1] <- exp(pars[3])   #cycle
  rho <- sigmoid(pars[7]) * 0.99
  per <- sigmoid(pars[8], 365, 1825)
  lam <- 2*pi/per
  vpsi <- model$Q[44, 44, 1] / (1 - rho^2)
  rho_co <- rho*cos(lam)
  rho_si <- rho*sin(lam)
  model$T[44:45, 44:45, 1] <- c(rho_co, -rho_si,
                                rho_si, rho_co)
  model$P1inf[44, 44] <- model$P1inf[45, 45] <- 0
  model$P1[44, 44] <- model$P1[45, 45] <- vpsi
  model$H[1, 1, 1] <- exp(pars[6])
  model
}

# Estimate
fit3 <- fitSSM(mod3, init, updt3, control = list(maxit = 2000))


# Smoothing
smo3 <- KFS(fit3$model, smoothing = c("state", "signal"), maxiter=200)


smo3_seas <- rowSums(smo3$alphahat[, seq(23,62,2)])
plot(y_train, type = "l")
lines(smo3$alphahat[, "level"], col = "red")      #Trend

lines(smo3$alphahat[, "level"] +
        smo3$alphahat[, "cycle"], col = "yellow") #Trend + Stagionalità intra-annua

lines(smo3$alphahat[, "level"] +
        smo3_seas, col = "blue")                  #Trend + Stagionalità intra-annua

plot(y_train, type = "l")
lines(smo3$alphahat[, "level"] +
        smo3$alphahat[, "sea_dummy1"] +
        smo3$alphahat[, "cycle"] +                #Trend + Stagionalità intra-annua +
        smo3_seas, col = "green")                 #Stag settimanale + ciclo

# Let us produce one-step-ahead predictions using the model estimated
# on the train set

Xtot <- as.data.frame(dum["2010-01-01/"])
y_tot <- c(y_train, rep(NA, length(y_val)))
Xtot$y <- as.numeric(y_tot)

est_3<- exp(fit3$optim.out$par)

mod33 <- SSModel(y ~ 0 + 
                   Dec24 +
                   Dec25 +
                   Dec26 +
                   Jan1 +
                   Jan6 +
                   Holidays +
                   HolySat +
                   HolySun +
                   EasterSat +
                   EasterSun +
                   EasterMon +
                   Aug15 +
                   BridgeDay +
                   EndYear +
                   SSMtrend(2, list(est_3[1], est_3[2])) +
                   SSMcycle(60, est_3[3]) +
                   SSMseasonal(7, est_3[4], "dummy") +
                   SSMseasonal(365, est_3[5], "trig",
                               harmonics = 1:20),
                 H = est_3[6],
                 data = Xtot)

# let us fix again the initial conditions so that they are not diffuse
mod33$a1 <- mod3$a1
mod33$P1 <- mod3$P1
mod33$P1inf <- mod3$P1inf

# Let us get smoother and signal filter (it includes one step ahead predictions)
smo33 <- KFS(mod33, smoothing = "state", filter = "signal")
one_step_ahead_3 <- xts(smo33$m[, 1], time(y["2010-01-01/"]))

plot(y["2017-03-14/"])
lines(one_step_ahead_3["2017-03-14/"], col = "red")

# Compute RMSE
sqrt(mean((one_step_ahead_3["2017-03-14/"] - y["2017-03-14/"])^2))

# Compute MAPE
mean(abs(one_step_ahead_3["2017-03-14/"] - y["2017-03-14/"]) / y["2017-03-14/"]) * 100


### MODELLO 4 ####
# LLT + CICLO STOCASTICO +  TREND GIORNALIERO SETTIMANALE (DUMMY) STOCASTICO 
# + TREND GIORNALIERO ANNUALE (TRIG) DETERMINISTICO

mod4 <- SSModel(y_train ~ 0 + 
                  Dec24 +
                  Dec25 +
                  Dec26 +
                  Jan1 +
                  Jan6 +
                  Holidays +
                  HolySat +
                  HolySun +
                  EasterSat +
                  EasterSun +
                  EasterMon +
                  Aug15 +
                  BridgeDay +
                  EndYear +
                  SSMtrend(2, list(NA, NA))  +
                  SSMcycle(1825, NA) +
                  SSMseasonal(7, NA, "dummy") +
                  SSMseasonal(365, 0, "trig", harmonics = 1:20),
                H = NA,
                data = Xtrain)

# We want to avoid diffuse initial conditions on the state variables
# Let us fix their means and variances to reasonable values
vary <- var(y_train)
mod4$a1["level", 1] <- mean(y_train)
mod4$P1inf <- mod4$P1inf * 0
diag(mod4$P1) <- vary


# Initial values for the variances we have to estimate
init <- numeric(7)
init[1] <- log(vary/10)  # log-var(err. slope)
init[2] <- log(vary/100)  # log-var(err. slope)
init[3] <- log(vary/10)   # log-var(err. cycle)
init[4] <- log(vary/100)  # log-var(err. dummy seas)
init[5] <- log(vary/100)  # log-var(err.oss)
init[6] <- 3.5            # sigmoid^(-1)(rho) (damping factor)
init[7] <- 0              # sigmoid^(-1)(period, 24, 96)

sigmoid <- function(x, a = 0, b = 1) {
  a + (b - a)/(1 + exp(-x))
}

updt4 <- function(pars, model) {
  model$Q[1, 1, 1] <- exp(pars[1])                           #intercept
  model$Q[2, 2, 1] <- exp(pars[2])                           #slope
  model$Q[3, 3, 1] <- exp(pars[4])                           #seas dummy
  model$Q[44, 44, 1] <- model$Q[45, 45, 1] <- exp(pars[3])   #cycle
  rho <- sigmoid(pars[6]) * 0.99
  per <- sigmoid(pars[7], 365, 1825)
  lam <- 2*pi/per
  vpsi <- model$Q[44, 44, 1] / (1 - rho^2)
  rho_co <- rho*cos(lam)
  rho_si <- rho*sin(lam)
  model$T[44:45, 44:45, 1] <- c(rho_co, -rho_si,
                                rho_si, rho_co)
  model$P1inf[44, 44] <- model$P1inf[45, 45] <- 0
  model$P1[44, 44] <- model$P1[45, 45] <- vpsi
  model$H[1, 1, 1] <- exp(pars[5])
  model
}

# Estimate
fit4 <- fitSSM(mod4, init, updt4, control = list(maxit = 2000))


# Smoothing
smo4 <- KFS(fit4$model, smoothing = c("state", "signal"), maxiter=200)


smo4_seas <- rowSums(smo4$alphahat[, seq(23,62,2)])
plot(y_train, type = "l")
lines(smo4$alphahat[, "level"], col = "red")      #Trend

lines(smo4$alphahat[, "level"] +
        smo4$alphahat[, "cycle"], col = "yellow") #Trend + Stagionalità intra-annua

lines(smo4$alphahat[, "level"] +
        smo4_seas, col = "blue")                  #Trend + Stagionalità intra-annua

plot(y_train, type = "l")
lines(smo4$alphahat[, "level"] +
        smo4$alphahat[, "sea_dummy1"] +
        smo4$alphahat[, "cycle"] +                #Trend + Stagionalità intra-annua +
        smo4_seas, col = "green")                 #Stag settimanale + ciclo

# Let us produce one-step-ahead predictions using the model estimated
# on the train set

Xtot <- as.data.frame(dum["2010-01-01/"])
y_tot <- c(y_train, rep(NA, length(y_val)))
Xtot$y <- as.numeric(y_tot)

est_4<- exp(fit4$optim.out$par)

mod44 <- SSModel(y ~ 0 + 
                   Dec24 +
                   Dec25 +
                   Dec26 +
                   Jan1 +
                   Jan6 +
                   Holidays +
                   HolySat +
                   HolySun +
                   EasterSat +
                   EasterSun +
                   EasterMon +
                   Aug15 +
                   BridgeDay +
                   EndYear +
                   SSMtrend(2, list(est_4[1], est_4[2])) +
                   SSMcycle(60, est_4[3]) +
                   SSMseasonal(7, est_4[4], "dummy") +
                   SSMseasonal(365, 0, "trig",
                               harmonics = 1:20),
                 H = est_4[5],
                 data = Xtot)

# let us fix again the initial conditions so that they are not diffuse
mod44$a1 <- mod4$a1
mod44$P1 <- mod4$P1
mod44$P1inf <- mod4$P1inf

# Let us get smoother and signal filter (it includes one step ahead predictions)
smo44 <- KFS(mod44, smoothing = "state", filter = "signal")
one_step_ahead_4 <- xts(smo44$m[, 1], time(y["2010-01-01/"]))

plot(y["2017-03-14/"])
lines(one_step_ahead_4["2017-03-14/"], col = "red")

# Compute RMSE
sqrt(mean((one_step_ahead_4["2017-03-14/"] - y["2017-03-14/"])^2))

# Compute MAPE
mean(abs(one_step_ahead_4["2017-03-14/"] - y["2017-03-14/"]) / y["2017-03-14/"]) * 100


