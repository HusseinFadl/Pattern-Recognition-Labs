import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("StudentAnswers.csv", sep=',',header=None)
fileNumpyMatrix = df.values

correctAnswers = ['a','d','e','c','a','d','d','e']
questionNames = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']

studentResponses = fileNumpyMatrix[:,1:]
print(studentResponses)
studentGender = fileNumpyMatrix[: , 0]
print(studentGender)


#Q1: TODO calculate number of students who didn't get the correct answer  for the second question Using for loops
#(1) using for loop

#(2) using vectorized code



#Q2: TODO calculate the average score of the students assuming that each correct answer gives 1 mark else 0
#(1) using for loop

#(2) using vectorized code



#Q3: TODO create a bar plot for the number students who answered the correct answer for each question



#Q4: TODO create pie plot for gender distribution of the students
