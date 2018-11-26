for i = 1:100
	str = ["SLIKA" num2str(i) ".txt"];
	data = dlmread(str, " ", 0, 0);
	str_output = ["SLIKA" num2str(i) ".png"];
	imwrite(data, str_output, "Quality", 100)
endfor
