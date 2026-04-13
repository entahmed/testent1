import pandas as pd 
age =pd.Series([22, 25, 30, 35, 40])
print(age)

studentInfo = pd.DataFrame({
'name': ['Sama', 'ahmed', 'Omer'], # String
'age': [21,22,23],

'isGraduated': [True, True, False]})
print("#" * 20)
print(studentInfo.index[0])
studentInfo = studentInfo.dropna(how='all')  
print(studentInfo[[]])


