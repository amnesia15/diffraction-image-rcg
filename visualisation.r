library(ggplot2)

x_value <- c()
y_value <- c()

for (i in 1:100) {
  x_value <- c(x_value, as.double(PARAM[[i]][1]))
  y_value <- c(y_value, as.double(PARAM[[i]][2]))
}

ggplot(mapping = aes(x = x_value, y = y_value)) + geom_point(shape = 18, color = "blue") + 
  xlab("R (nm)") + ylab("H (nm)")

mean_value <- c()
for (i in 1:251) {
  mean_value <- c(mean_value, mean(DIFSLIKA[[i]]$V3))
}

ggplot(mapping = aes(x = x_value, y = y_value)) + geom_tile(aes(fill = mean_value)) +
  scale_fill_gradient(low = "white", high = "black") + geom_abline(slope = 2, intercept = 0)

write.table(x = mean_value, file = "Desktop/mean_value", row.names = FALSE)

x_unique <- unique(x_value)
y_unique <- unique(y_value)
z_matrix <- array(, dim = c(length(unique(x_value)), length(unique(y_value))))

for (i in 1:251) {
  i_ind <- which(x_value[i] == x_unique)
  j_ind <- which(y_value[i] == y_unique)
  
  z_matrix[i_ind, j_ind] <- mean(DIFSLIKA[[i]]$V3)
}

write.table(z_matrix, file = "mean_matrix.txt", row.names = FALSE, col.names = FALSE)
write.table(unique(x_value), file = "x_value.txt", row.names = FALSE, col.names = FALSE)
write.table(unique(y_value), file = "y_value.txt", row.names = FALSE, col.names = FALSE)

na_rows <- c()
na_cols <- c()
na_count <- 0
for (i in 1:nrow(z_matrix)) {
  for (j in 1:ncol(z_matrix)) {
    if (is.na(z_matrix[i, j])) {
      na_rows <- c(na_rows, i)
      na_cols <- c(na_cols, j)
      na_count <- na_count + 1
    }
  }
}

na_rows <- sort(unique(na_rows[1:100]))
na_cols <- sort(unique(na_cols[1:100]))

color_value <-  y_value / x_value

color_value <- sapply(color_value, function(x) ifelse(x >= 2, 1, 0))

ggplot(mapping = aes(x = x_value, y = y_value, color = color_value)) + 
  geom_point() + geom_abline(slope = 99/40, intercept = -2375) + scale_x_continuous(limits = c(0, 5000))
