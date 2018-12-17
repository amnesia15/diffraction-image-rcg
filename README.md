# Diffraction Image Parameter Recognition

Za transformaciju podataka dobijih od strane fortran programa koristi se skripta **transform_data.py**. Skripta ima sledece parametre:
* **-dp** - putanja do ulaznih podataka (podrazumevano **images/raw_data/**)
* **-op** - putanja za izlazne slike (podrazumevano **images/**)
```
python transform_data.py -dp images/raw_data/ -op images/
```

Za kreiranje .png slika od matrice intenziteta koristi se **create_image.m**.
```
octave create_image.m
```

Za treniranje neuronske mreze nad podacima koristi se **training_model.py** skripta. Skripta ima sledece parametre i to:
* **-ip** - putanja do ulaznih slika (podrazumevano **"images/"**)
* **-pp** - putanja do ulaznih parametara slike (R, H) (podrazumevano **"images/params/"**)
* **-e** - broj epoha za treniranje neuronske mreze (podrazumevano **6000**)
* **-lr** - learning rate za optimizator (podrazumevano **0.00001**)
* **-i** - broj treniranja jedne iste arhitekture (podrazumevano **1**)
* **-hl** - broj jedinica po skrivenom sloju (podrazumevano **[108, 55]**)
* **-mo** - putanja za snimanja modela, grafika i predikcija (podrazumevano **"model_output/"**)
* **-ts** - procenat podataka koji ce biti sacuvan kao skup za testiranje, broj izmedju 0.0 i 1.0 (podrazumevano **0.2**)
* **-bs** - broj uzorka po azuriranju gradijenta (podrazumevano **32**)
* **-r** - da li koristiti regularizaciju (0 - bez ikakve regularizacije, 1 - dropout metoda) (podrazumevano **0**)
* **-dr** - dropout rates po skrivenom sloju (podrazumevano **[0.3, 0.1]**)

Nijedan parameter nije obavezan ako ste zadovoljni podrazumevanim vrednostima.

Jedno pokretanje training_model.py izgleda:
```
python training_model.py -ip images/ -pp images/params/ -e 6000 -lr 0.00001 -i 1 -hl 108 55 -mo model_output/ -ts 0.2 -bs 32 -r 1 -dr 0.3 0.1
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
* **-ts** - procenat podataka koji ce biti sacuvan kao skup za testiranje, broj izmedju 0.0 i 1.0 (podrazumevano **0.2**)
* **-bs** - broj uzorka po azuriranju gradijenta (podrazumevano **32**)
* **-r** - da li koristiti regularizaciju (0 - bez ikakve regularizacije, 1 - dropout metoda) (podrazumevano **0**)
* **-dr** - dropout rates po skrivenom sloju (podrazumevano **[0.3, 0.1]**)

Jedno pokretanje za isprobavanje razlicitih arhitektura izgleda:
```
python generate_architecture.py -ip images/ -pp images/params/ -e 50 -lr 0.0001 -mo model_output/ -ul 90 110 40 60 -ts 0.2 -bs 32 -r 1 -dr 0.3 0.1
```

**loading.r** is used for loading files into R environment, normalizing images, and creating matrix corresponding to grayscale image pixels.  

**create_image.m** is used for generating images. This script should be used from directory of the previously saved files.
