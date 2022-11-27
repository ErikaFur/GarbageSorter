# GarbageSorter


### How to use classification model?  

1. Install requirements from GarbageSorter/requirements.txt
2. Place your image in folder GarbageSorter/processing/toSort
3. Run GarbageSorter/main.ipynb from GarbageSorter directory (it contains relative pathes)
4. Cell with main() can be run multiple times unil folder toSort is not empty
5. Predicted class will appear in the console as prediction will be over.
6. Logs appear in GarbageSorter/processing/logGarbage.csv after each prediction. Each record contains: name of processed image, class number and class name
7. All processed images moved from folder GarbageSorter/processing/toSort in the folder GarbageSorter/processing/sorted
