# ECE 8803 Term Project: Comparison between Four Classification Methods for DRSS Severity Classification on OCT images 
## by Tawhid Khan & Syed Anas Hussain

This Repository contains code for 4 classifications methods we used: K-Nearest Neighbors, Logistic Regression, Convolutional Neural Network, and Random Forest


## Downloading Dataset

Download and extract the "Prime_FULL" dataset from https://zenodo.org/record/7105232 

## K-Nearest Neighbors

1. Make sure dataset is in same directory as  the "knn.py" file:

    ```console
    .
    ├── ...
    
    ├── Downloads 
          ├── Prime_FULL                
          ├── knn.py 
    ├── ...
    ```
         
 
2. Open command line and navigate to the directory where the dataset and code is such as "Downloads" directory in this example. 

3. Run the following command and specify count of neighbors k such as "5":

    ```console
    python knn.py --number_neighbors 5
    ```
  
4. The program will output to terminal various performance metrics of the model:

   ```console
   (base) PS C:\Users\tawhid khan\downloads> python knn.py --number_neighbors 5
    Data split finished
    knn classifier created
    knn classifier fitted
    predictions generated on test dataset
    Accuracy: 0.39
    Balanced Accuracy: 0.34
    Precision: 0.38
    Recall: 0.39
    F1 Score: 0.39
    True Positive Rate: 0.77
    False Positive Rate: 0.83
    ```

## Logistic Regression

1. Make sure dataset is in same directory as  the "logreg.py" file:

    ```console
    .
    ├── ...
    
    ├── Downloads 
          ├── Prime_FULL                
          ├── logreg.py 
    ├── ...
    ```
         
 
2. Open command line and navigate to the directory where the dataset and code is such as "Downloads" directory in this example. 

3. Run the following command and specify maximum number of iterations in algorithm such as "1500":

    ```console
    python logreg.py --epochs 1000
    ```
  
4. The program will output to terminal various performance metrics of the model:

    ```console
   (base) PS C:\Users\tawhid khan\Downloads> python logreg.py --epochs 1500 
    Data split finished
    Logistic Regression classifier created
    Logistic Regression classifier fitted
    predictions generated on test dataset
    Accuracy: 0.38
    Balanced Accuracy: 0.31
    Precision: 0.37
    Recall: 0.38
    F1 Score: 0.37
    True Positive Rate: 0.70
    False Positive Rate: 1.52
    ```

   
