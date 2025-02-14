for file in a1a a2a a3a # a4a a5a a6a a7a a8a a9a australian_scale.txt fourclass_scale.txt german.numer_scale gisette_scale.bz2 gisette_scale.t.bz2 heart_scale.txt ijcnn1.bz2 ionosphere_scale.txt leu.bz2 liver-disorders_scale.txt madelon.txt mushrooms.txt phishing.txt skin_nonskin.txt sonar_scale.txt splice_scale.txt svmguide1.txt svmguide3.txt w1a.txt w2a.txt w3a.txt w4a.txt w5a.txt w6a.txt w7a.txt w8a.txt
do
 	echo "Running $file"
	python exp_logistic.py --dataset problems/$file --tol 1e-04 --max_iter 1000
done
