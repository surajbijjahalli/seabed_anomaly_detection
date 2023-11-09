#! /bin/bash




file_list=$(cd config && ls)

#echo $file_list



for file in $file_list
    do python Main.py $file
        
done
