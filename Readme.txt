# Face_Recognition
Facial Recognition using Bayesian Classifier, KNN, Kernel SVM, Boosted SVM, PCA, and MDA

## How to run the code
To run the code keep your data in ./Data/ folder and use the following command:

pyhon <filename>.py --task_id <task_id> --transform <transform_name> --data_name <data_name>

Given the filename and the other input arguments the default parameters will load and the you can obtain the 
results as mentioned in the report. filename can take ```bayes, knn, kernel_svm, boosted_svm``` as its values.
task_id can take ```1 or 2``` based on the task you want to do. Task_id 1 is identifying the label and 
task_id 2 is neutral vs expression classification. transform_name can take ```PCA or MDA```. data_name can
take ```data, pose, illum``` for task1 and just ```data``` for task2. Some of the example commands to run the 
code are:

python knn.py --transform PCA --task_id 1 --data_name data
python kernel_svm.py --kernel rbf --transform MDA
python boosted_svm.py --transform MDA

Note: If you want to plot the plots then uncomment the plotting code in each .py file and manually
      create the folder path in your system
