for i in `seq 41 79`
do
	python oracle2tree_drs.py dev.tree.align_drs.oracle.doc.in $i > $i.tree
done
