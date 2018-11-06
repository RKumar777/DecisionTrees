# DecisionTrees
Building a Decision tree in Python
The goal here is to build a decision tree from scratch and without using pandas package
"sys.argv" command takes in the different inputs/outputs required for the program
The different inputs/outputs here are:
1. input .csv file containing the different features and the last column as the output column (training set)
2. input .csv file containing the different features and the last column as the output column (test set)
3. depth of the tree (hyperparameter)
4. Output file for train file predictions
5. Output file for test file predictions
6. Output file for train and test error

When depth=0, the classifier becomes a majority vote classifier and directly calculates the result by counting the output which occurs more

When depth>0, splitting is done through information gain criterion following the ID3 algorithm
The function 'entcal' is used for calculation of entropy for each of the features as well as the output
The function 'mutinf' calculates mutual information between output and the different features and then return the information gain
The function 'dectree' takes in an array and gives out the decision tree
The function 'noebuild' is used to build the different nodes of the decision tree
