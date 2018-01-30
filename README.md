



```
echo review_path,label > data.csv; find -name *.txt | awk -v OFS=',' '{if ($1 ~ /pos/) {print $1, 0} else {print $1, 1}}' >> data.csv
```