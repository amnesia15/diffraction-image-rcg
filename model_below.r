library(imager)

data_images <- array(, dim = c(1000, 201, 201))

for (i in 1:1000) {
  str_path <- paste0("Desktop/1000_imgs/SLIKA", i, ".png")
  im <- load.image(str_path)
  data_images[i, , ] <- im[1:201, 1:201]
}

data_labels <- array(, dim = c(1000, 2))
data_labels <- INPUT


data_x <- array(, dim = c(1000, 101))
for (i in 1:1000) {
  data_x[i, ] <- data_images[i, 101, 101:201]
}



set.seed(12)
training_indices <- sample(seq_len(251), size = round(251 * 0.8))
train_data <- data_x[training_indices, ]
test_data <- data_x[-training_indices, ]
train_labels <- data_labels[training_indices, ]
test_labels <- data_labels[-training_indices, ]


library(keras)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 3000, activation = 'relu', input_shape = c(101)) %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1500, activation = 'relu') %>%
  #layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1500, activation = 'relu') %>%
  #layer_dropout(rate = 0.3) %>%
  layer_dense(units = 750, activation = 'relu') %>%
  #layer_dropout(rate = 0.2) %>%
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
  epochs = 100,
  validation_split = 0.2
)

model %>% evaluate(test_data, test_labels)

predicted <- model %>% predict(test_data)
predicted <- cbind(predicted, test_labels)
