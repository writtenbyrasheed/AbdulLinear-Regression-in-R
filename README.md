# Linear Regression Internship Project

## CS239 Demo by Abdul Bakare.

### Setup: INCD V VECI

1. Sign up for an account on Posit Cloud
2. Create a Workspace and name it workspace2
3. Create a project and name it p1
4. Reach the R studio console
5. Start your work at the terminal

### Linear Regression:
Linear regression on a dataset called Advertising.csv.

Download from the reference:https://www.kaggle.com/code/natigmamishov/linear-regression-from-scratch-on-advertising-data/input



Upload from the Downloaded Advertising.scv file through the upload table under files.


Type  file through the upload tab under Files


Upload the downloaded Advertising.csv file through the upload table under files.


###Import data into R STUDIO
Type `advert <- read.csv("/cloud/project/Advertising.csv") into the console


## examine the top few rows of the data
```
head(advert)
```

![Alt](https://github.com/abdul/AbdulLinear-Regression-in-R/images/summary.txt)



#View the data and generate the summary
```


View(advert)
summary(advert)
```
![Alt](https://github.com/abdul/AbdulLinear-Regression-in-R/images/summary.txt)

##Install packages ISLR, MASS, ggplot.multistats and import library ggplot2

```
install.packages("ISLR")
install.packages("MASS")
install.packages("ggplot2")
library(ggplot2)
```
--------------------
###Examining the relationship between the TV budget and Sales using scatterplots

```
ggplot(advert, aes(TV, Sales)) + geom_smooth(method="lm") +
geom_point() + ggtitle("linear fit")
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/ScatterplotTvBudgetSales.png.txt)

## Use lm() function to create (and estimates) a linear model with the TV budget data and sales data

```
lm.TV = lm(Sales~TV, data=advert)
```


## Create a dataframe for new values of TV budgets to be able to use predict( function)

```
TV_budget <- data.frame(TV = c(11,11,12,12,12,12,13,13,13,13))
```
##predict the Sales

```
predict(lm.TV, newdata = TV_budget)
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/salespredictions.txt)

-----------------------

###Examining the relationship between the Newspaper budget and Sales using scatterplots

```
ggplot(advert, aes(Newspaper, Sales)) + geom_smooth(method="lm") +
geom_point() + ggtitle("linear fit")
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/ScatterplotNewspaperBudgetSales.png.txt)

## Use lm() function to create (and estimates) a linear model with the Newspaper budget data and sales data

```
lm.Newspaper = lm(Sales~Newspaper, data=advert)
```


## Create a dataframe for new values of Newspaper budgets to be able to use predict( function)

```
Newspaper_budget <- data.frame(Newspaper = c(11,11,12,12,12,12,13,13,13,13))
```
##predict the Sales

```
predict(lm.Newspaper, newdata = Newspaper_budget)
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/salespredictions.txt)

-----------------------

###Examining the relationship between the Radio budget and Sales using scatterplots

```
ggplot(advert, aes(Radio, Sales)) + geom_smooth(method="lm") +
geom_point() + ggtitle("linear fit")
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/ScatterplotRadioBudgetSales.png.txt)

## Use lm() function to create (and estimates) a linear model with the Radio budget data and sales data

```
lm.Radio = lm(Sales~Radio, data=advert)
```


## Create a dataframe for new values of Radio budgets to be able to use predict( function)

```
Radio_budget <- data.frame(Radio = c(11,11,12,12,12,12,13,13,13,13))
```
##predict the Sales

```
predict(lm.Radio, newdata = Radio_budget)
```
![Alt](https://github.com/writtenbyrasheed/AbdulLinear-Regression-in-R/images/salespredictions.txt)


First we will discover the data available within the data package. In the console, type data() to see a list of the available datasets available within the data package.

Type `data()` into the console.

We will be working with the cars dataset. We will first load the data and see what it contains:

```
data(cars)
View(cars)
```
![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/cars.png)

We now want to visualize the data to see if we can get an understanding of how it is structured.

`plot(dist~speed, data=cars)`

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/2dcar_plot.png)

Looking at the data, it seems that there seems to be a linear relationship between the speed and stopping distance of cars. For a quantitative measure of how well the data correlates, we can check the pearson and spearman correlation of the data:

```
cor(cars, use="complete.obs", method="pearson")
cor(cars, use="complete.obs", method="spearman")
```

We obtain a pearson correlation factors of 0.8068949 and a spearman correlation factor of 0.8303568. This is high enough to indicate that the variables are indeed related in some fashion. Now that we are convinced there is a relationship between the data, we can use the speed of a car to predict its stopping distance. For our first model, we will train a model of the form Y = β<sub>1</sub> + β<sub>2</sub>X + ε where Y is the car breaking distance and X is the car's speed. To train a linear model on the data, we use the `lm()` command:

`model <- lm(dist~speed, data=cars)`
We now have a trained linear model that predicts the stopping distance of a car given its speed. From the output of the model, we can also see our regression line: Distance = -17.58 + 3.93 * Speed. To visualize our regression line, we can overlay it with the original training data. 

`abline(model, col="red")`

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/2dcar_model_plot.png?raw=true)

To view additional details of the model, use the `summary()` command:

`summary(model)`

The output should look something like this:

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/lm_call.png?raw=true)

The summary provides lots of data on the model such as the R squared and adjusted R squared values, the F statistic, and the p-value and is a valuable tool for evaluating the model. Since p < 0.05, we can reject the null hypothesis. We can also view other information such as the error sum of squares and mean sum of squares through the `anova()` command.

`anova(model)`

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/anova.png?raw=true)

Additionally, we can graphically analyze the statistical properties of our model.

```
plot(model)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/residualvsfitted.png?raw=true)

To demonstrate the capabilities of R, we can also easily calculate relevant values on our own. For instance the following code will produce the SSE, SST, estimated variance, R^2, and adjusted R^2 values. Note that we can use our model to make predictions using the `predict()` command.

```
prediction <- predict(model, cars)
#can get error sum of squares SSE:
SSE <- sum((prediction - cars$dist)^2)

#estimated variance
variance <- SSE/(n-2)

#can get total sum of squares SST
meanDist <- mean(cars$dist)
SST <- sum((cars$dist - meanDist) ^2)

#get coefficient of determination R^2
r2 <- 1 - (SSE/SST)

#get adjusted R^2
r2adj <- ((n-1)*r2 - k) / (n-1-k)
```

Now let's move on to something slightly more complicated. If we look back at our plot, it appears that there may be an exponential rather than linear relationship with speed. To address this, we can synthesize an additional independent variable (speed^2) and use that for predictions.
First, we will extend our dataset:

```
cars2 <- cars
cars2["speed^2"] <- cars$speed^2
View(cars2)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/cars2.png?raw=true)

Now we can train our model using different training and test sets. The following lines of code randomly sample the data we have and splits it into a training set with 80% of the data and a test set with 20% of the data.

```
set.seed(100)
trainingIndex <- sample(1:nrow(cars), 0.8*nrow(cars))
cars2_train <- cars2[trainingIndex, ]
cars2_test <- cars2[-trainingIndex, ]
```

Training the model is as easy as before!

```
model2 <- lm(dist ~ speed + `speed^2`, data=cars2_train)
```

Our model now predicts the breaking distance using the speed and the squared speed. If we want to visualize this data, we will need to create a 3d plot. First, we will need to install and load the required 3d scatterplot package:

```
install.packages("scatterplot3d")
library("scatterplot3d")
```
To visualize the data in 3d:

```
cars2plot <- scatterplot3d(cars2_train$speed, cars2_train$`speed^2`, cars2_train$dist, angle=45)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/3dcar_plot.png?raw=true)

We can also draw our regression plane:

```
cars2plot$plane3d(model2)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/3dcar_model_plot.png?raw=true)

We can view further information about our new model as before with the `summary()` and `anove()` commands. Predicitions are also made similarly with the `predict()` command.

### A Little Step-wise and Logistic Regression
This section is intended to get you introduced to step-wise and logistic regression in R. Many of the same commands used in linear regression are applicable here so we will not go over them again. Instead, this will be a brief introduction to get you up and running.

To introduce step-wise regression, we will use a slightly more complex dataset - the state.x77 dataset included with R. We will load the data in as a dataframe.

```
state=as.data.frame(state.x77)
View(state)
cor(state)
step(lm(`Life Exp`~Murder, data=state), direction="both", scope=~Population+Income+Murder+Illiteracy+Area+Frost+`HS Grad`)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/state.png?raw=true)

The work is done with the `step()` command which evaluates models based on the Akaike Information Criterion (AIC) which measures the likelihood of a model and thus can be used for model evaluation. We begin the process by checking the correlation between all independent variables and the dependent variable, Life Expectancy. We include in the scope of the search all fields given to us. The `step()` command will then add and remove parameters in search of a model with the lowest AIC score. The output of the above step command can be seen here:

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/stepwise.png?raw=true)

In terms of code, logistic regression is very similar to linear regression. We first select our data - in this case, none of the builtin datasets were suitable so we use one online: (http://www.ats.ucla.edu/stat/data/binary.csv). We then use the `glm()` command to train our model and the rest is analysis! One point to note is that the rank column needs to be converted to a categorical variable to be treated properly.

```
admissions <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
View(admissions)
admissions$rank <- factor(admissions$rank)
admissionsModel <- glm(admit ~ gre + gpa + rank, data = admissions, family = "binomial")
summary(admissionsModel)
```

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/admission.png?raw=true)

![Alt](https://github.com/andrewc94/Linear-Regression-in-R/raw/master/images/logistic.png?raw=true)

### References
(http://r-statistics.co/Linear-Regression.html)

(http://www.ats.ucla.edu/stat/r/dae/logit.htm)
