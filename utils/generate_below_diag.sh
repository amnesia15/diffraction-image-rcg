#!/bin/bash

r_beg=1000
r_end=5000

h_beg=100
h_end=10000

interval1=40
interval2=50

r_leap=$(( r_end - r_beg ))
h_leap=$(( h_end - h_beg ))

r_leap=$( echo "scale=3; $r_leap * 1.0" | bc)
h_leap=$( echo "scale=3; $h_leap * 1.0" | bc)

r_leap=$( echo "scale=3; $r_leap / ($interval1 - 1)" | bc)
h_leap=$( echo "scale=3; $h_leap / ($interval2 - 1)" | bc)

echo "r_leap = $r_leap h_leap = $h_leap"

r_iter=$r_beg
h_iter=$h_beg
no_iter=1

while [ "$(echo "$r_iter <= $r_end" | bc)" -eq 1 ]; do
    while [ "$(echo "$h_iter <= $h_end" | bc)" -eq 1 ]; do
    	test_var=$( echo "$r_iter * 99/40 - 2375" | bc )
    	
    	if [ "$(echo "$h_iter <= $test_var" | bc)" -eq 1 ]
		then
			h_print=$( echo $h_iter | awk '{print int($1+0.5)}' )
		    r_print=$( echo $r_iter | awk '{print int($1+0.5)}' )
		       
		    mkdir "input$no_iter"
		    echo $r_print > "input$no_iter/INPUT.TXT"
		    echo $h_print >> "input$no_iter/INPUT.TXT"
		    
		    no_iter=$(( $no_iter + 1 ))
		fi
		
        h_iter=$( echo "scale=3; $h_iter + $h_leap" | bc )
    done
    r_iter=$( echo "scale=3; $r_iter + $r_leap" | bc)
    h_iter=$h_beg
done
