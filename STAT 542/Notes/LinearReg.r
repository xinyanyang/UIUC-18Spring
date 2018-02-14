# STAT / CES 542, Fall 2018
# This is part I of the R code for lecture note LinearReg

# lets first generate some data:

# generate 100 observations with 3 variables
set.seed(1)
n = 1000
p = 500
x = matrix(rnorm(n*p), n, p)
X = cbind(1, x) # the design matrix, including 1 as the first column

# define the true beta, the first entry is the intercept
b = as.matrix(c(1, 1, 0.5, rep(0, p-2))) 

# generate training y with Gaussian errors
y = X %*% b + rnorm(n)

# fit a linear regression model 
lm.fit = lm(y ~ x)

# look at the coefficients beta hat
lm.fit$coef

# different ways to solve for the beta estimations
# !!! These are not necessarily efficient algorithms, only for demonstration purpose
# Try a larger n and p

# using normal equations by inverting the X'X matrix: b = (X'X)^-1 X'y 
?solve
system.time({beta_hat = solve(t(X) %*% X) %*% t(X) %*% y})
beta_hat

# or solve the linear equation system X'X b = X'y 
system.time({beta_hat = solve(t(X) %*% X, t(X) %*% y)})
beta_hat

# QR decomposition
# direct calling the qr.coef function
system.time({beta_hat = qr.coef(qr(X), y)})
beta_hat

# or 
system.time({beta_hat = qr.solve(t(X) %*% X, t(X) %*% y)})
beta_hat

# if you want to see what Q and R are
QR = qr(X)
Q = qr.Q(QR)
R = qr.R(QR)

# get inverse of R, you can check R %*% R_inv yourself
# the backsolve/forwardsolve functions can be used to solve AX = b for upper/lower triangular matrix A 
?backsolve
R_inv = backsolve(R, diag(p+1), upper.tri = TRUE, transpose = FALSE)
beta_hat = R_inv %*% t(Q) %*% y

# Cholesky Decomposition 

# the chol function gives upper triangular matrix instead of lower
# crossprod(X) is X'X
R = t(chol(crossprod(X)))
w = forwardsolve(R, t(X) %*% y, upper.tri = FALSE, transpose = FALSE)
beta_hat = forwardsolve(R, w, upper.tri = FALSE, transpose = TRUE) # the transpose = TRUE means that we are solving for R'b = w instead of Rb = w 

# or equivalently 
system.time({
R = chol(crossprod(X))
w = backsolve(R, t(X) %*% y, upper.tri = TRUE, transpose = TRUE)
beta_hat = backsolve(R, w, upper.tri = TRUE, transpose = FALSE)
})


# fit linear regression to the ozone data 
# we need this package to get the data
library(ElemStatLearn)

# fit the full model with all three covariates: radiation, temperature and wind
fullmodel = lm(ozone~ radiation + temperature + wind, data=ozone)
round(summary(fullmodel)$coef, dig=3)

# get the residual wind after projecting on other covariates (including intercept)
wind.res = lm(wind ~ radiation + temperature, data=ozone)$res

# now, wind.res is orthogonal to all other covariates, 
# We can regression y on it to get the original coefficient for wind
# '-1' means we are excluding the intercept
parmodel=lm(ozone ~ -1 + wind.res, data=ozone)
round(summary(parmodel)$coef, dig=3)

# which is the same as finding the project
sum(wind.res*ozone$ozone)/sum(wind.res*wind.res)


# some other things about fitting a linear model using lm()
# these are not necessarily 

# you can add an interaction terms by specifying them
lm.fit = lm(ozone~ radiation + temperature + wind + wind*temperature, data=ozone)
round(summary(lm.fit)$coef, dig=3)

# this creates all two-way and three-way interactions of the specified terms
lm.fit = lm(ozone~ radiation + temperature + wind + wind*temperature*radiation, data=ozone)
round(summary(lm.fit)$coef, dig=3)

# alternatively, if you want to include all terms, use '.' to represent them
lm.fit = lm(ozone~ ., data=ozone) # all first order 
lm.fit = lm(ozone~ .*., data=ozone) # all first and second order 
lm.fit = lm(ozone~ .*.*., data=ozone) # up to third order interactions
round(summary(lm.fit)$coef, dig=3)



# linear model/variable selection


# We analyze the Diabetes Data (Efron et al, 2004) using different model selection criteria and algorithms 

# Get the Diabetes Data from the package "lars"
library(lars)
data(diabetes)
diab = data.frame(cbind(diabetes$x, "Y" = diabetes$y))

# A Brief Description of the Diabetes Data (Efron et al, 2004):
# Ten baseline variables: age, sex, body mass index, average blood pressure, 
# and six blood serum measurements were obtained for each of n = 442 
# diabetes patients, as well as the response of interest, a quantitative 
# measure of disease progression one year after baseline 

pairs(diab, pch=".")  # produce pair-wise scatter plots. Caution: a big figure.

# a fancier plot, requires another package
library(PerformanceAnalytics)
suppressWarnings(chart.Correlation(diab, col = "purple", pch = "*"))

#####################################################
lmfit=lm(Y~., data=diab)
names(lmfit)  # What have been returned by "lm"?
lmfit$coef    # 12 regression coefficients including the intercept

summary(lmfit)
summary(lmfit)$coef  # coefficients and the corresponding p-values

# our definition for calculating the AIC if use normal distribution likelihood, 12 is the number of parameters
n = nrow(diab)
p = 11

?AIC
AIC(lmfit) # a build-in function for calculating AIC using -2log likelihood
n*log(sum(residuals(lmfit)^2/n)) + n + n*log(2*pi) + 2 + 2*p

# In many standard R packages, the AIC is calculated by removing some constants from the likelihood 
# We will use this value as the default
?extractAIC
extractAIC(lmfit) # AIC for the full model
RSS = sum(residuals(lmfit)^2)
n*log(RSS/n) + 2*p

# so the BIC for the full model is 
extractAIC(lmfit, k = log(n))
n*log(RSS/n) + log(n)*p

#####################################################
# Model selection: stepwise algorithm 

?step

step(lmfit, direction="both")            # AIC
step(lmfit, direction="both", trace=0)   # do not print intermediate results

step(lmfit, direction="backward")
step(lm(Y~1, data=diab), scope=list(upper=lmfit, lower=~1), direction="forward")

step(lmfit, direction="both", k=log(n))  # BIC (the default value for k=2, which corresponds to AIC)


##########################################################################
# Best subset model selection (Cp, AIC, and BIC): leaps 
##########################################################################
library(leaps)

# performs an exhaustive search over models, and gives back the best model 
# (with low RSS) of each size.
# the default maximum model size is nvmax=8

RSSleaps=regsubsets(as.matrix(diab[,-11]),diab[,11])
summary(RSSleaps, matrix=T)

RSSleaps=regsubsets(as.matrix(diab[,-11]),diab[,11], nvmax=10)
summary(RSSleaps,matrix=T)

sumleaps=summary(RSSleaps,matrix=T)
names(sumleaps)  # components returned by summary(RSSleaps)

sumleaps$which
msize=apply(sumleaps$which,1,sum)
n=dim(diab)[1]
p=dim(diab)[2]
Cp = sumleaps$rss/(summary(lmfit)$sigma^2) + 2*msize - n;
AIC = n*log(sumleaps$rss/n) + 2*msize;
BIC = n*log(sumleaps$rss/n) + msize*log(n);

cbind(Cp, sumleaps$cp)
cbind(BIC, sumleaps$bic)  # It seems regsubsets uses a formula for BIC different from the one we used. 
BIC-sumleaps$bic  # But the two just differ by a constant, so won't affect the model selection result. 
n*log(sum((diab[,11] - mean(diab[,11]))^2/n)) # the difference is the score of an intercept model

# Rescale Cp, AIC, BIC to (0,1).
inrange <- function(x) { (x - min(x)) / (max(x) - min(x)) }

Cp = sumleaps$cp; Cp = inrange(Cp);
BIC = sumleaps$bic; BIC = inrange(BIC);
AIC = n*log(sumleaps$rss/n) + 2*msize; AIC = inrange(AIC);


plot(range(msize), c(0, 1.1), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize, Cp, col="red", type="b")
points(msize, AIC, col="blue", type="b")
points(msize, BIC, col="black", type="b")
legend("topright", lty=rep(1,3), col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))

# zoom in
id=3:7;
plot(range(msize[id]), c(0, 0.25), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize[id], Cp[id], col="red", type="b")
points(msize[id], AIC[id], col="blue", type="b")
points(msize[id], BIC[id], col="black", type="b")
legend("topright", lty=rep(1,3), col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))

# Who's the 2nd best (regarding BIC)?
RSSleaps=regsubsets(as.matrix(diab[,-11]),diab[,11], nbest=4, nvmax=10)
sumleaps=summary(RSSleaps,matrix=T)
sumleaps
msize=apply(sumleaps$which,1,sum);
Cp = sumleaps$cp; Cp = inrange(Cp);
BIC = sumleaps$bic; BIC = inrange(BIC);
AIC = n*log(sumleaps$rss/n) + 2*msize; AIC = inrange(AIC);

plot(range(msize), c(0,1.1), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize, Cp, col="red")
points(msize, AIC, col="blue")
points(msize, BIC, col="black")
legend("topright", lty=rep(1,3), col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))

id=(1:length(msize))[msize %in% 5:8]; 
plot(range(msize[id]), c(0, 0.1), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize[id], Cp[id], col="red", cex=2, pch=1)
points(msize[id], AIC[id], col="blue", cex=2, pch=2)
points(msize[id], BIC[id], col="black", cex=2, pch=3)
legend("topright", pch=1:3, col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))

id=(1:length(msize))[msize %in% 5:8]; myloc=rnorm(length(id))*0.1
plot(range(msize[id]), c(0, 0.1), type="n", xlab="Model Size (with Intercept)", ylab="Model Selection Criteria")
points(msize[id]+myloc, Cp[id], col="red", cex=2, pch=1)
points(msize[id]+myloc, AIC[id], col="blue", cex=2, pch=2)
points(msize[id]+myloc, BIC[id], col="black", cex=2, pch=3)
legend("topright", pch=1:3, col=c("red", "blue", "black"), legend=c("Cp", "AIC", "BIC"))

# How to retrieve the variable subet from the best model returned by Cp, AIC, and BIC 
varid=sumleaps$which[order(Cp)[1],]
varid
names(diab)[varid[-1]]

varid=sumleaps$which[order(AIC)[1],]
names(diab)[varid[-1]]

varid=sumleaps$which[order(BIC)[1],]
names(diab)[varid[-1]]
sumleaps$which[order(BIC)[1:3],]
sumleaps$bic[order(BIC)[1:3]] # the top (smallest) two BIC are very close




















