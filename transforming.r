#loading data

DIFSLIKA <- vector("list", 1000)

for (i in 1:1000) {
  str_path = paste0("Desktop/1000/input", i, "/DIFSLIKA.TXT")
  temp_data_frame <- read.table(str_path, quote = "\"", comment.char = "")
  DIFSLIKA[[i]] <- temp_data_frame
}

#loading input parameters

INPUT <- array(, dim = c(1000, 2))

for (i in 1:1000) {
  str_path = paste0("Desktop/1000/input", i, "/INPUT.TXT")
  res <- readLines(str_path)
  INPUT[i, 1] <- as.double(res[1])
  INPUT[i, 2] <- as.double(res[2])
}

# finding max luminance

max_lum <- max(DIFSLIKA[[1]]$V3)

for (i in 2:1000) {
  max_cur <- max(DIFSLIKA[[i]]$V3)
  if (max_cur > max_lum)
    max_lum <- max_cur
}

# normalizing

for (i in 1:1000) {
  DIFSLIKA[[i]]$V3 <- DIFSLIKA[[i]]$V3 / max_lum
}

# transforming indexes

for (i in 1:1000) {
  DIFSLIKA[[i]]$V1 <- (DIFSLIKA[[i]]$V1 + 10000) / 100 + 1
  DIFSLIKA[[i]]$V2 <- (DIFSLIKA[[i]]$V2 + 10000) / 100 + 1
}


# creating matrices 

IMG_MAT <- array(, dim = c(1000, 201, 201))

for (i in 1:1000) {
  for (j in 1:nrow(DIFSLIKA[[i]])) {
    IMG_MAT[i, DIFSLIKA[[i]][j, 1], DIFSLIKA[[i]][j, 2]] <- DIFSLIKA[[i]][j, 3]
  }
}

# writing matrices to files

for (i in 1:1000) {
  str_path <- paste0("Desktop/1000_imgs/SLIKA", i, ".txt")
  write.table(IMG_MAT[i, ,], file = str_path, row.names = FALSE, col.names = FALSE)
}