import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("StudentAnswers.csv", sep=',',header=None)
fileNumpyMatrix = df.values

correctAnswers = ['a','d','e','c','a','d','d','e']
questionNames = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']

#TODO calculate number of students who didn't get the correct answer  for the second question Using for loops
studentResponses = fileNumpyMatrix[:,1:]
print(studentResponses)
studentGender = fileNumpyMatrix[: , 0]
print(studentGender)

#using for loop
counter = 0
for studentAnswer in studentResponses :
    if studentAnswer[1]!= correctAnswers[1]:
        counter += 1
print(counter)

#using vectorized code

print(sum((studentResponses[:,1] != correctAnswers[1]).astype('int')))

#TODO calculate the average score of the students assuming that each correct answer gives 1 mark else 0

#using for loop
scoreAvg =0
for studentAnswer in studentResponses:
    sScore = 0
    for questionIndex in range(len(studentAnswer)):
        if(studentAnswer[questionIndex]==correctAnswers[questionIndex]):
            sScore+= 1
    scoreAvg += sScore

scoreAvg/=studentResponses.shape[0]
print( scoreAvg)

#using vectorized code
print(np.average(np.sum((studentResponses==np.array(correctAnswers)).astype('int'), axis=1) ))


#TODO create a bar plot for the number students who answered the correct answer for each question
correctCount = np.sum((studentResponses==np.array(correctAnswers)).astype('int'), axis=0)

plt.bar(questionNames,correctCount,color='#157880')
plt.show()

#TODO create pie plot for gender distribution of the students

genders , genderCount= np.unique(studentGender , return_counts=True)
plt.pie(genderCount , labels=genders)
plt.show()