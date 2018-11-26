#!/usr/bin/env python

import matplotlib.pyplot as plt

X = []
f = open('x_value.txt', 'r')
lines = f.readlines()
f.close()

for num in lines:
	X.append(int(num))
	
Y = []
f = open('y_value.txt', 'r')
lines = f.readlines()
f.close()

for num in lines:
	Y.append(int(num))
	
Z = []
f = open('mean_matrix.txt', 'r')
for i in range(0, 10):
	new_row = []
	lines = f.readline().split(' ')
	for num in lines:
		new_row.append(float(num))
	Z.append(new_row)
f.close()
	
plt.contourf(X, Y, Z, 100)
plt.xlabel('R (nm)')
plt.ylabel('H (nm)')
plt.colorbar()
plt.title('Contour plot (mean value)')
plt.savefig('color_plot.png')