# GarbageSorter


### How to use classification model?  

1. Place your image in folder GarbageSorter/processing/toSort
2. Run cell GarbageSorter/main.ipynb
3. Cell with main() can be run multiple times unil folder toSort is not empty
4. Predicted class will appear in the console as prediction will be over.
4. Logs appear in GarbageSorter/processing/logGarbage.csv after each prediction. Each record contains: name of processed image, class number and class name
5. All processed images moved from folder GarbageSorter/processing/toSort in the folder GarbageSorter/processing/sorted
