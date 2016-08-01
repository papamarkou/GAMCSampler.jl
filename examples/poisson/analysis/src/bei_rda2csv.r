# Exmaple taken from http://www.petrkeil.com/?p=1709

# Load required libraries
library(sp)
library(raster)

# Set environment variables
DATADIR <- "../../data"

# Load data
load(file.path(DATADIR, "bei.rda"))

# Visualize tree locations
plot(
  bei$x,
  bei$y,
  pch=20,
  cex=0.5,
  col="darkgreen",
  main="Location of trees in the 50-ha Barro Colorado plot", 
  xlab="x coordinate (m)",
  ylab="y coordinate (m)",
  frame=FALSE
)

# Define elevation and gradient predictors
elevation <- raster(bei.extra[[1]])
gradient <- raster(bei.extra[[2]])

# Crop the predictors so that they have exactly 500 x 1000 cells
ext <- extent(2.5, 1002.5, 2.5, 1002.5)
elevation <- crop(elevation, ext)
gradient <- crop(gradient, ext)

# Create coarse grid of predictors into 50 x 50 m grid by taking the mean of the 5 x 5 m grid cells
elevation50 <- aggregate(elevation, fact=10, fun=mean)
gradient50 <- aggregate(gradient, fact=10, fun=mean)

# Fit spatial data into the 50 x 50 m grid
xy <- data.frame(x = bei$x, y = bei$y)

# Create coarse response
# The following two vectors coincide:
# rasterize(xy, gradient50, fun = "count")[]
# rasterize(xy, gradient50, fun = "count")[]
n50 <- rasterize(xy, elevation50, fun="count")
n50[is.na(n50)] <- 0 # Replace NA values by 0

# Visualization of coarse data
plot(elevation50, main="Predictor: mean elevation in 50x50 m cells")
plot(gradient50, main="Predictor: mean gradient in 50x50 m cells")
plot(n50, main="Response: # of Individuals in 50x50 m cells")

# Standardize predictors
standardize <- function(x) {
  (x - mean(x))/sd(x)
}

sdelevation50 <- standardize(elevation50[])

sdgradient50 <- standardize(gradient50[])

sqsdelevation50 <- sdgradient50^2

# Run Poisson regression
poissonreg <- glm(n50[] ~ sdelevation50+sqsdelevation50+sdgradient50, family="poisson")

summary(poissonreg)

write.csv(
  data.frame(
    n50=n50[],
    elevation50=elevation50[],
    gradient50=gradient50[]
  ),
  file=file.path(DATADIR, "coarse_bei.csv"),
  quote=FALSE,
  row.names=FALSE
)
