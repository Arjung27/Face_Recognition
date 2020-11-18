python bayes.py --data_name data
python bayes.py --data_name pose
python bayes.py --data_name illum

python bayes.py --data_name data --transform MDA
python bayes.py --data_name pose --transform MDA
python bayes.py --data_name illum --transform MDA

python bayes.py --data_name data --task_id 2
python bayes.py --data_name pose --task_id 2
python bayes.py --data_name illum --task_id 2

python bayes.py --data_name data --transform MDA --task_id 2
python bayes.py --data_name pose --transform MDA --task_id 2
python bayes.py --data_name illum --transform MDA --task_id 2

python knn.py --data_name data
python knn.py --data_name pose
python knn.py --data_name illum

python knn.py --data_name data --transform MDA
python knn.py --data_name pose --transform MDA
python knn.py --data_name illum --transform MDA

python knn.py --data_name data --task_id 2
python knn.py --data_name pose --task_id 2
python knn.py --data_name illum --task_id 2

python knn.py --data_name data --transform MDA --task_id 2
python knn.py --data_name pose --transform MDA --task_id 2
python knn.py --data_name illum --transform MDA --task_id 2
