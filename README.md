# Project 2 Readme

ALL FILES REQUIRE vocabulary.txt, newsgrouplables.txt to be run. 

data preparing:
    categorization.py is the file the loads in the training data and the testing data. These csv files are large enough that its better to read the data in chunks and write them to a csr_matrix, as well as stack those matrices on top of each other for one final csr_matrix of the training data and the testing data, respectively. This will output a file called "[type of data]_sparse.npz"
    Because of the chunk size syntax of python, it misses the last row of the testing and training data. So, load.py takes in those npz files created by categorization.py, and outputs another npz file called "final_[type of data]_sparse.npz". this file will be read into the main programs for naive bayes and logistic regression.
    I will add in these files to the submission along with the "make" files of these file, but I would not suggest running them, and just use the npz files premade when running the classifiers.

model 1 (Naive Bayes):
    There are a few parts to the project necessary for naive bayes classifier: training for a given beta, training for a set of beta, training for the best beta and giving a confusion matrix. Each of these requires commenting or uncommenting certain lines of code in the main file called load_check.py.
    
    Training for a given beta / best beta:
        For this, you can see at line 60, I can set beta to be 1/abs_V. Since this was just to start the implementation, it is not necessary to run the code with that beta. However, if you like, you can in line 78 write "for B in [beta]:" to see the code run with beta in place. Follow the instructions for "best beta", to run this and output a prediction file titled submission.csv. (instructions to follow this example)
        Training for best_beta:
            Once you run the set of beta and find the highest probability, you can run this with Beta = [best_beta] that produced the highest probability.
            Running this step:
                I have labeled these with BB for best beta.
                The lines to uncomment in loadcheck.py are 76-77, 148-156, 200-205.
                
    Training for a set of beta:
        This was to answer question 2 after initially trianing beta, we added in a set of beta to compare to accuracy on the validation set. In this case, the for loop at line 78 can be set to "for B in Beta". Where Beta is a list of candidate betas
            Running this step:
                I have labeled all sections that are required to be used to run this part of the code with the label TS (Training for the Set of beta).
                The lines to uncomment in loadcheck.py are 74, 135-142, 182-192 (leave line 181 commented, it has two ## to ensure if you run control+/ on those lines it will stay commented.)
                
    Training for the confusion matrix:
        This was to answer quetion 3 for naive bayes, to make a confusion matrix for the best beta found from the set of beta investigation. Because you are making this with the test data from the training data (validation set, it is a combination of some of the previous two methods.).
            Running this step:
                These are labeled with CM.
                The lines to uncomment in loadcheck.py are 76-77, 135-142, 161-171, and 210 (if you want to save to a file the confusion matrix for naive bayes.)
                
    Quick note, the lines metioned above should be switched off (commented out) when not in use. Also, run best_beta instructions to create the submission file for kaggle.

Model 2 (logistic regression):
    There are a couple parts in this portion of the project as well: training of the classifier for different lambda, eta, and iteration number (called step_limit in the code, unfortunately) and making a confusion matrix.
    
    Training for sets of the hyperparameters:
        This is a simpler set of instructions than for naive bayes. We can run this for a step_limit which is really iteration numbers, of 500,1000,5000, etc. To check for code running, i would run 500, so it wont take as long.
        In addition you can make the etal or the lambal as long or short as you like.
        To run without adaptive hyperparameters, comment out lines 136-144. Otherwise eta and lambda will change once an error threshold is reached.
        You'll notice i divided the training data by 1000, for eta and lambda at a max of 0.015 each, this is unnecessary, we just got nicer/quicker results with that added factor and higher eta/lambda
        The submission file will print with lines 170-192.
        To make the tables of eta and lambda comparisons, you'll want to uncomment lines 149-153, and lines 122-133
        
    Making a confusion matrix.
        You'll want this to run with only one eta, one lambda, and one step_limit.
        Use lines 201-231 for this to run.
        
Entropy for question 6 and printing out top 100 words:
    This is located in the entropy.py file
    This takes in premade files called: 'yes_frame_entropy_problem6.npz', and 'no_frame_entropy_problem6.npz'
    No frame was premade using lines 84-109. The lines 87-98 make the class fractions for entropy calculation of the main training set. Subtract the yes frame from that gives you the no frame.
    Yes frame was premade in the info_gain.py file. 
