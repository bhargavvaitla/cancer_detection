
import csv
import math
import statistics
def load_from_csv(filePath):
    """
    Loads data from a given file 
   
    Parameters
    ----------
    filePath : str
        Location of the CSV file 

    Returns
    -------
    List
        returs a data read from the file location in matrix format(List of list)

    """
    data = []
    with open(filePath) as dataFile:
        dataReader = csv.reader(dataFile);
        for row in dataReader:
            tempRow = [int(x) for x in row]
            data.append(tempRow);
    
    return data;


def get_distance(data1,data2):
    """
    Calculates the Euclidean distance between two points 

    Parameters
    ----------
    data1 : List
            Data point 1.
    data2 : List
            Data point 2.

    Returns
    -------
    float
        returns the euclidean distance between data1 and data2.

    """
    return math.sqrt(sum([(a-b) **2 for a,b in zip(data1,data2)]))
        

def get_standard_deviation(data,i):
    """
    Calculates the standard deviation of a given column in matrix 

    Parameters
    ----------
    data : list
        observed data of patients in matrix form.
    i : int
        coulumn number in matrix.

    Returns
    -------
    float
        returns SD of a column i in matrix data.

    """
    n = len(data);
    #retrieved i th element from every list (which makes a ith column in matrix format)
    column = [a[i-1] for a in data];
    mean = sum(column)/len(column);
    
    #List that contains square of diffrence element and mean of that perticular column
    diff = [(a-mean) ** 2 for a in column];
    std = math.sqrt(sum(diff)/(n-1));
    
    return std;

def get_standardised_matrix(data):
    """
    It transform the matrix data into standardized form

    Parameters
    ----------
    data : list
        matrix data that needs to standardized.

    Returns
    -------
    list
        returns the standardized form of matrix data.

    """
    rows = len(data);
    columns = len(data[0]);
    mean = [];
    std = [];
    stdMatrix= [];
    #Calculates mean and standard deviation of every column and append them to 
    #mean and std lists respectively
    for i in range(columns):
        column = [a[i] for a in data];
        mean.append(sum(column)/len(column));
        std.append(get_standard_deviation(data,i+1));
    #using the mean and std lists , standardized the matrix
    for i in range(rows):
        stdRow = [];
        for j in range(columns):
            stdVal =(data[i][j] - mean[j])/std[j];
            stdRow.append(stdVal);
        stdMatrix.append(stdRow);
        
    
    return stdMatrix;

def get_k_nearest_labels(rowData, learningData , labels, k):
    """
    It identifies the K nearest neighbours(observations) and 
    returns the outcomes of those observations

    Parameters
    ----------
    rowData : list
        a single observation whose K nearest neighbours to be found.
    learningData : list
        Learning data or training data of KNN algorithm in matrix form .
    labels : list
        matrix that contains outcomes of training data.
    k : int
        number of neighbours to be considered.

    Returns
    -------
    list
        returs a matrix that contain K neighbour outcome labels .

    """
    data_distance =[];
    kNearLabels = [];
    #calculates the distance of every element of learningData from the rowData
    #append the distance along with corresponding label of learningData into a single list 
    for i in range(len(learningData)):
        learningData_distance = [get_distance(rowData,learningData[i])];
        learningData_distance.append(learningData[i]);
        learningData_distance.append(labels[i]);
        data_distance.append(learningData_distance);
    
    
    #sort the list based on the distance
    data_distance.sort(key= lambda a:a[0]);
    
    #fetch the first K elements from the list(K nearest elements) and 
    #return the corresponding labels of that elements 
    for i in range(k):
        kNearLabels.append(data_distance[i][-1]);
    
    return kNearLabels;


def get_mode(KNearLabels):
    """
    It returns the mode of list of elements 

    Parameters
    ----------
    KNearLabels : list
        matrix containing labels of k nearest neighbours.

    Returns
    -------
    int
        returns an element with highest frequency in KNearLabels.

    """
    labels = [a[0] for a in KNearLabels]
    return statistics.mode(labels);



def classify(data , learningData,learningDataLabels, k):
    """
    It classfies the observations using the knowledge obtained throught stored learning data  

    Parameters
    ----------
    data : list
           a list of observations whose outcome need to be predicted (in matrix form).
    learningData : list
        a list of strored observations (Matrix form) whose experience is used in making prediction 
        for unseen data.
    learningDataLabels : list
         outcomes for stored observations (learningData) in matrix form.
    k : int
        number of neighbouring observations to be considered.

    Returns
    -------
    list
        returns the predicted outcomes for dataset data.

    """
    data_labels =[];
    
    for i in range(len(data)):
        kNearLabels = get_k_nearest_labels(data[i],learningData,learningDataLabels,k);
        data_labels.append([get_mode(kNearLabels)]);
    
    return data_labels;
        

def get_accuracy(correct_data_labels,data_labels):
    """
    Calculates accuracy of predictions 

    Parameters
    ----------
    correct_data_labels : list
        correct outcomes  of dataset.
    data_labels : TYPE
        predicted outcomes of dataset.

    Returns
    -------
    float
        returns accuracy of predictions.

    """
    counter = 0 ;
   
    for i in range(len(data_labels)):
       if data_labels[i][0] == correct_data_labels[i][0]:
           counter += 1;
    
    return (counter/len(data_labels)) * 100.0;
        
        

def run_test():
    """
    It loads all the data sets needed and print the accuracy of predicted outcomes with
    various K ranging from 3 to 15

    Returns
    -------
    None.

    """
    learning_data = load_from_csv('C:\PC\python_assignment\data\Learning_Data.csv');
    data = load_from_csv('C:\PC\python_assignment\data\Data.csv');
    correct_data_labels = load_from_csv('C:\PC\python_assignment\data\Correct_Data_Labels.csv');
    learning_data_labels = load_from_csv('C:\PC\python_assignment\data\Learning_Data_Labels.csv');
    
    #standardizes the matrix learning_data and data 
    learning_data = get_standardised_matrix(learning_data);
    data = get_standardised_matrix(data);
    
    #print the accuracy with every K from 3 to 15
    for i in range(3,16):
        data_labels = classify(data,learning_data,learning_data_labels,i);
        acc = get_accuracy(correct_data_labels,data_labels);
        print('K= '+ str(i)+ ' Accuracy= '+str(acc));
    
    

        
    
    
    
    


def main():
    run_test();

if __name__ == '__main__':
    main()
