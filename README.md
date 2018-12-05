# diffraction-image-rcg


Za treniranje neuronske mreze nad podacima koristi se training_model.py skripta. Skripta ima podrazumevane parametre i to:
* -ip - putanja do ulaznih slika (podrazumevano "images/")
* -pp - putanja do ulaznih parametara slike (R, H) (podrazumevano "images/params/")
* -e - broj epoha za treniranje neuronske mreze (podrazumevano 6000)
* -lr - learning rate za optimizator (podrazumevano 0.00001)
* -i - broj treniranja jedne iste arhitekture (podrazumevano 1)
* -hl - broj jedinica po skrivenom sloju (podrazumevano [108, 55])
* -mo - putanja za snimanja modela, grafika i predikcija (podrazumevano "model_output/")

**loading.r** is used for loading files into R environment, normalizing images, and creating matrix corresponding to grayscale image pixels.  

**create_image.m** is used for generating images. This script should be used from directory of the previously saved files.

**model.r** is used for creating keras model.
