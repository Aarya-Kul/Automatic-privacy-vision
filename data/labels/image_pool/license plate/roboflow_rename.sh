#!/usr/bin/env bash

# script to rename image files downloaded from roboflow from
# e.g. 112480_jpg.rf.30257de1c11d1f09e9e67cf498e02797.jpg
# to 112480_rf.jpg.

files=(*_jpg.rf.*.jpg *_jpg.rf.*.txt)

for file in "${files[@]}"; do
  if [ -e "$file" ]; then
    base=$(echo "$file" | awk -F'_jpg' '{print $1}')
    extension="${file##*.}"
    newname="${base}_rf.${extension}"
    mv "$file" "$newname"
  fi
done
