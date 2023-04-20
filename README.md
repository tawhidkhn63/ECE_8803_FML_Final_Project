# ECE 8803 Term Project: Comparison between Four Classification Methods for DRSS Severity Classification on OCT images

This Repository contains code for 4 classifications methods we used: K-Nearest Neighbors, Logistic Regression, Convolutional Neural Network, and Random Forest


## Downloading Dataset

Download and extract the "Prime_FULL" dataset from https://zenodo.org/record/7105232 

## K-Nearest Neighbors

1. Make sure dataset is in same directory as  the "knn.py" file:
    .
    ├── ...
    ├── Downloads
          ├── Prime_FULL                
          ├── knn.py 
    ├── ...
         
 
2. Open command line and navigate to the directory where the dataset and code is such as "Downloads" directory in this example. 

3. Run the following command and specify count of neighbors k such as "5":

  ```console
  python knn.py --number_neighbors 5
  ```
  
4. The program will output to terminal various performance metrics of the model:

   ![image](https://user-images.githubusercontent.com/39498885/233450543-f0ca5dba-1997-419f-9f06-b194b425e925.png)
