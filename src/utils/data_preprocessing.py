from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def data_preprocessing(df):
    df.Age = df.Age.fillna(df.Age.mean())
    df = df.drop(columns=["Cabin"])
    df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)
    
    # we even don't need it
    df = df.drop(columns=["PassengerId"])
    
    mlb = MultiLabelBinarizer()
    Embarked_encoding = mlb.fit_transform(df['Embarked'])
    Embarked_df = pd.DataFrame(Embarked_encoding, columns=mlb.classes_)
    df = pd.concat([df.drop('Embarked', axis=1), Embarked_df], axis=1)
    
    # only for test dataset
    if True in df["Fare"].isnull().to_list():
        df.fillna({"Fare": df["Fare"].mean()}, inplace=True)
    
    #! Temporary
    df = df.drop(columns=["Name", "Ticket"])
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    return df