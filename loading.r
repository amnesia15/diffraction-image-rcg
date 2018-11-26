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

# finding min and max for luminance 
max_lum <- DIFSLIKA[[1]]$V3[1]
min_lum <- DIFSLIKA[[1]]$V3[1]
for (i in 1:251) {
  min_cur = min(DIFSLIKA[[i]]$V3)
  max_cur = max(DIFSLIKA[[i]]$V3)
  if (max_cur > max_lum) {
    max_lum <- max_cur
  }
  if (min_cur < min_lum) {
    min_lum <- min_cur
  }
}

# normalizing images
for (i in 1:251) {
  DIFSLIKA[[i]]$V3 <- DIFSLIKA[[i]]$V3 / max_lum
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

# checking the correctness of normalization
max_sv <- DIFSLIKA[[1]]$V3[1]
for (i in 1:251) {
  if (max(DIFSLIKA[[i]]$V3) > max_sv) {
    max_sv <- max(DIFSLIKA[[i]]$V3)
  }
}

max(DIFSLIKA[[5]]$V3)

image(IMG_MAT[[1]], useRaster = TRUE, axes = TRUE, col = gray((0:65536)/65536))


# loading INPUT files 
PARAM <- vector("list", 100)
for (i in 1:100) {
  str_path = paste("Desktop/master/input", i, "/INPUT.TXT", sep = "")
  res <- readLines(str_path)
  res[0] <- as.double(res[1])
  res[1] <- as.double(res[2])
  PARAM[[i]] <- res
}

for (i in 1:151) {
  str_path = paste("Desktop/new_cluster/input", i, "/INPUT.TXT", sep = "")
  res <- readLines(str_path)
  res[0] <- as.double(res[1])
  res[1] <- as.double(res[2])
  PARAM[[i + 100]] <- res
}



# finding diffraction image with max luminance
max_values_img <- c()
max_value_ind = 1

for (i in 1:100) {
  if (i > 1 && (max(DIFSLIKA[[i]]$V3) > max(max_values_img)))
    max_value_ind = i
  max_values_img <- c(max_values_img, max(DIFSLIKA[[i]]$V3))
}

# writing matrices to files

for (i in 1:251) {
  str_path <- paste("Desktop/image_matrix/SLIKA", i, ".txt", sep = "")
  write.table(IMG_MAT[[i]], file = str_path, row.names = FALSE, col.names = FALSE)
}
