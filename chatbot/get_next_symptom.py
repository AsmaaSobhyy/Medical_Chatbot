from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import pandas as pd




def get_next_symptom(old_symptoms_yes,old_symptoms_no):
  clean_df=pd.read_csv('clean_data.csv')
  selected= clean_df.copy()
  for sym in old_symptoms_yes:
    selected = selected[selected[sym]== 1]
  for sym in old_symptoms_no:
    selected = selected[selected[sym]== 0]
  X= selected.drop(['prognosis'],axis=1)
  Y = selected['prognosis']
  clf = DecisionTreeClassifier(random_state=0)
  clf = clf.fit(X,Y)

  text_representation = tree.export_text(clf,feature_names=clean_df.columns[:-1].tolist())
  #print(text_representation)
  if text_representation.split('\n')[0].split(' ')[1] == 'class:': 
    return None
  return text_representation.split('\n')[0].split(' ')[1]

# input: 
# old_symptoms_yes -> array containing the symptoms that the user have
# old_symptoms_no -> array containing the symptoms that the user doesn't have
# output : string containing the next symptom or none 
print(get_next_symptom(old_symptoms_yes=['skin_rash','itching'],old_symptoms_no=[]))
