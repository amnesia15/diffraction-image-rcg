# diffraction-image-rcg


Za treniranje neuronske mreze nad podacima koristi se **training_model.py** skripta. Skripta ima sledece parametre i to:
* **-ip** - putanja do ulaznih slika (podrazumevano **"images/"**)
* **-pp** - putanja do ulaznih parametara slike (R, H) (podrazumevano **"images/params/"**)
* **-e** - broj epoha za treniranje neuronske mreze (podrazumevano **6000**)
* **-lr** - learning rate za optimizator (podrazumevano **0.00001**)
* **-i** - broj treniranja jedne iste arhitekture (podrazumevano **1**)
* **-hl** - broj jedinica po skrivenom sloju (podrazumevano **[108, 55]**)
* **-mo** - putanja za snimanja modela, grafika i predikcija (podrazumevano **"model_output/"**)

Nijedan parameter nije obavezan ako ste zadovoljni podrazumevanim vrednostima.

Jedno pokretanje training_model.py izgleda:
```
python training_model.py -ip images/ -pp images/params/ -e 6000 -lr 0.00001 -i 1 -hl 108 55 -mo model_output/
```

Za vrsenje predikcije nad nekom slikom koristi se **predict.py** skripta. Skripta ima sledece parametre:
* **-p** - putanju do slike 

Jedno pokretranje za predikciju izgleda:
```
python predict.py -p images/SLIKA1.png
```

Za isprobavanje razlicitih kombinacija arhitekture koristi se skripta **generate_architecture.py**. Skripta ima sledece parametre: 
* **-ip** - putanja do ulaznih slika (podrazumevano **"images/"**)
* **-pp** - putanja do ulaznih parametara slike (R, H) (podrazumevano **"images/params/"**)
* **-e** - broj epoha za treniranje neuronske mreze (podrazumevano **50**)
* **-lr** - learning rate za optimizator (podrazumevano **0.00001**)
* **-mo** - putanja za snimanja modela, grafika i predikcija (podrazumevano **"model_output/"**)
* **-ul** - granice broja jedinica u skrivenom sloju, po dva broja za jedan sloj (donja i gornja granica ukljucujuci i ta oba broja) (podrazumevano **[90, 110, 40, 60]**)

Jedno pokretanje za isprobavanje razlicitih arhitektura izgleda:
```
python generate_architecture.py -ip images/ -pp images/params/ -e 50 -lr 0.0001 -mo model_output/ -ul 30 40
```

**loading.r** is used for loading files into R environment, normalizing images, and creating matrix corresponding to grayscale image pixels.  

**create_image.m** is used for generating images. This script should be used from directory of the previously saved files.
