The data for this project is located in the scratch folder of the pace clusters because of the memory limitations. The way I have it set up is like this: 
- cd ~/scratch
- mkdir /HyTech/ConeDetection/data/fsoco/
- cd /HyTech/ConeDetection/data/fsoco/
- curl -L -C - -o fsoco_boxes_train.tar "http://fsoco.cs.uni-freiburg.de/src/download_boxes_train.php"