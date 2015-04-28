#!/bin/bash
for i in $(ls $1.*);
do
    sed -i "s/$1/$2/Ig" $i
done
rename -v s/$1/$2/ $1.*
