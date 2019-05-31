for m in `seq 41 79`
do
	echo $m
	rm models/model
	ln -s model${m} models/model
	bash dev.sh ${m}
done
