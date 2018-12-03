library(ggplot2)

## Ploting predicted vs real R

ggplot(mapping = aes(x = test_labels[, 1], y = predicted[, 1])) + geom_point() + geom_abline(slope = 1) +
  xlim(500, 5500) + ylim(500, 5500) + xlab("R (nm)") + ylab("Predicted R (nm)") +
  ggtitle("Predicted vs real R")

## Ploting predicted vs real H

ggplot(mapping = aes(x = test_labels[, 2], y = predicted[, 2])) + geom_point() + geom_abline(slope = 1) + 
  xlim(0, 10500) + ylim(0, 10500) + xlab("H (nm)") + ylab("Predicted H (nm)") +
  ggtitle("Predicted vs real H")


library(Metrics)

# Calculating mean absolute error for R

mae(test_labels[, 1], predicted[, 1])

# Calculating mean absolute error for H

mae(test_labels[, 2], predicted[, 2])

# Calculating mean of mean absolute errors

(mae(test_labels[, 1], predicted[, 1]) + mae(test_labels[, 2], predicted[, 2])) / 2


# Finding points above and below diagonal line 

above_diag = c()

for (i in 1:nrow(test_labels)) {
  if ((test_labels[i, 1] * 99/40 - 2375) > test_labels[i, 2])
    above_diag <- c(above_diag, 1)
  else 
    above_diag <- c(above_diag, 0)
}

above_diag <- as.factor(above_diag)

# Ploting points

ggplot(mapping = aes(x = test_labels[, 1], y = test_labels[, 2], color = above_diag)) + geom_point() + 
  geom_abline(slope = 99/40, intercept = -2375) + ggtitle("Test set real values") + xlab("R") + ylab("H")

# Finding points above and below diagonal

points_above_diag <- c()
predicted_above_diag <- c()
points_below_diag <- c()
predicted_below_diag <- c()


for (i in 1:length(above_diag)) {
  if (above_diag[i] == 0) {
    points_above_diag <- rbind(points_above_diag, test_labels[i, ])
    predicted_above_diag <- rbind(predicted_above_diag, predicted[i, ])
  }
  else {
    points_below_diag <- rbind(points_below_diag, test_labels[i, ])
    predicted_below_diag <- rbind(predicted_below_diag, predicted[i, ])
  }
}

# Calculating mean absolute error for R (above diagonal)

mae(points_above_diag[, 1], predicted_above_diag[, 1])

# Calculating mean absolute error for H (above diagonal)

mae(points_above_diag[, 2], predicted_above_diag[, 2])

# Calculating mean absolute error for R (below diagonal)

mae(points_below_diag[, 1], predicted_below_diag[, 1])

# Calculating mean absolute error for H (below diagonal)

mae(points_below_diag[, 2], predicted_below_diag[, 2])


## Ploting predicted vs real R (above diagonal)

ggplot(mapping = aes(x = points_above_diag[, 1], y = predicted_above_diag[, 1])) + geom_point() + geom_abline(slope = 1) +
  xlim(500, 5500) + ylim(500, 5500) + xlab("R (nm)") + ylab("Predicted R (nm)") +
  ggtitle("Predicted vs real R (above diagonal)")

## Ploting predicted vs real H (above diagonal)

ggplot(mapping = aes(x = points_above_diag[, 2], y = predicted_above_diag[, 2])) + geom_point() + geom_abline(slope = 1) +
  xlim(0, 10500) + ylim(0, 10500) + xlab("H (nm)") + ylab("Predicted H (nm)") +
  ggtitle("Predicted vs real H (above diagonal)")

## Ploting predicted vs real R (below diagonal)

ggplot(mapping = aes(x = points_below_diag[, 1], y = predicted_below_diag[, 1])) + geom_point() + geom_abline(slope = 1) +
  xlim(500, 5500) + ylim(500, 5500) + xlab("R (nm)") + ylab("Predicted R (nm)") +
  ggtitle("Predicted vs real R (below diagonal)")

## Ploting predicted vs real H (below diagonal)

ggplot(mapping = aes(x = points_below_diag[, 2], y = predicted_below_diag[, 2])) + geom_point() + geom_abline(slope = 1) +
  xlim(0, 10500) + ylim(0, 10500) + xlab("H (nm)") + ylab("Predicted H (nm)") +
  ggtitle("Predicted vs real H (below diagonal)")
