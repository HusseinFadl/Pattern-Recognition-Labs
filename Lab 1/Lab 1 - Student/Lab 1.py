import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("StudentAnswers.csv", sep=',',header=None)
fileNumpyMatrix = df.values

correctAnswers = ['a','d','e','c','a','d','d','e']
questionNames = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']

#TODO calculate number of students who didn't get the correct answer  for the second question Using for loops and vectorized code
# Hint: to convert boolean arrays to integers you can use "astype" function and it is used as follows
# np.array([True , True , False]).astype('int') ---> [ 1, 1 , 0 ]


#using for loop


#using vectorized code


#TODO calculate the average score of the students assuming that each correct answer gives 1 mark else 0
# Hint: to caclulate the average value of a numpy array you can use "np.average" function and it is used as follows
# np.average(np.array([2 ,3, 4 ])) ---> 3



#using for loop


#using vectorized code



#TODO create a bar plot for the number students who answered the correct answer for each question


#TODO create pie plot for gender distribution of the students

