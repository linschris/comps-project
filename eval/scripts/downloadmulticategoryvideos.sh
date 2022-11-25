# Check number of arguments
if [ "$#" != "2" ]; then
	echo "Usage: bash downloadmulticategoryvideos.sh <number-of-videos-per-category> <selected-category-file-name>"
	exit 1
fi

line = "Cat"

bash downloadcategoryids.sh $1 "${line}"
bash downloadvideos.sh $1 "${line}"