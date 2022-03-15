# the submission column looks odd, trying to fix the [[]] structure.
import pandas as pd

submission = pd.read_csv('submission.csv')
print(submission['class'])

#found here https://www.geeksforgeeks.org/python-extract-characters-except-of-k-string/
counter_string = '[[]]'
for j in range(0,len(submission)):
    res = []
    for ele in submission.iloc[j,1]:
        if ele not in counter_string:
            res.append(ele)
            
    res = ''.join(res)
    res = int(res)
    submission.iloc[j,1] = res
    
print(submission)
submission.to_csv('submission_fixed.csv', index=False)
