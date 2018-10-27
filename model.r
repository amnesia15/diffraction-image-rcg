library(imager)

data_images <- array(, dim = c(256, 201, 201))

for (i in 1:251) {
  str_path <- paste0("Desktop/image_matrix/SLIKA", i, ".png")
  im <- load.image(str_path)
  data_images[i, , ] <- im[1:201, 1:201]
}

data_labels <- array(, dim = c(251, 2))

for (i in 1:100) {
  str_path = paste("Desktop/master/input", i, "/INPUT.TXT", sep = "")
  res <- readLines(str_path)
  data_labels[i, 1] <- as.double(res[1])
  data_labels[i, 2] <- as.double(res[2])
}

for (i in 1:151) {
  str_path = paste("Desktop/new_cluster/input", i, "/INPUT.TXT", sep = "")
  res <- readLines(str_path)
  data_labels[i + 100, 1] <- as.double(res[1])
  data_labels[i + 100, 2] <- as.double(res[2])
}



set.seed(12)
training_indices <- sample(seq_len(251), size = round(251 * 0.8))
train_data <- data_images[training_indices, , ]
test_data <- data_images[-training_indices, , ]
train_labels <- data_labels[training_indices, ]
test_labels <- data_labels[-training_indices, ]

train_data <- array_reshape(train_data, c(nrow(train_data), 201 * 201))
test_data <- array_reshape(test_data, c(nrow(test_data), 201 * 201))


library(keras)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(201 * 201)) %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 2)

summary(model)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_absolute_error")
)

history <- model %>% fit(
  train_data,
  train_labels,
  epochs = 30,
  validation_split = 0.2
)
