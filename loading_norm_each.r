# loading DIFSLIKAx.TXT files
DIFSLIKA <- vector("list", 251)

for (i in 1:100) {
  str_path = paste("Desktop/master/input", i, "/DIFSLIKA.TXT", sep = "")
  temp_data_frame <- read.table(str_path, quote="\"", comment.char="")
  DIFSLIKA[[i]] <- temp_data_frame
}

for (i in 1:151) {
  str_path = paste0("Desktop/new_cluster/input", i, "/DIFSLIKA.TXT")
  temp_data_frame <- read.table(str_path, quote = "\"", comment.char = "")
  DIFSLIKA[[i + 100]] <- temp_data_frame
}


# normalization
for (i in 1:251) {
  max_val <- max(DIFSLIKA[[i]]$V3)
  DIFSLIKA[[i]]$V3 <- DIFSLIKA[[i]]$V3 / max_val
}

# transforming indexes
for (i in 1:251) {
  DIFSLIKA[[i]]$V1 <- (DIFSLIKA[[i]]$V1 + 10000) / 100 + 1
  DIFSLIKA[[i]]$V2 <- (DIFSLIKA[[i]]$V2 + 10000) / 100 + 1
}

# creating matrices 
IMG_MAT <- vector("list", 251)

for (i in 1:251) {
  IMG_MAT[[i]] <- matrix(, 201, 201)
  for (j in 1:nrow(DIFSLIKA[[i]])) {
    IMG_MAT[[i]][DIFSLIKA[[i]][j, 1], DIFSLIKA[[i]][j, 2]] <- DIFSLIKA[[i]][j, 3]
  } 
}


# load images 
data_images <- array(, dim = c(251, 201, 201))

for (i in 1:251) {
  data_images[i, , ] <- IMG_MAT[[i]]
}

data_x <- array(, dim = c(251, 102))
for (i in 1:251) {
  data_x[i, 1:101] <- data_images[i, 101, 101:201]
}

for (i in 1:251) {
  
  if (data_x[i, 1] > data_x[i, 2])
    count_zero_diff <- 1
  else
    count_zero_diff <- 0
  
  for (j in 2:100) {
    if (data_x[i, j] > data_x[i, j - 1] && data_x[i, j] > data_x[i, j + 1])
      count_zero_diff <- count_zero_diff + 1
  }
  
  data_x[i, 102] <- count_zero_diff
}
