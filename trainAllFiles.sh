#!/bin/bash
FILES=System_Images/*
for file in $FILES
do
  echo "Training from $file file."
  ./openCV_Detect_Fruit $file t
done
