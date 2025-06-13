from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path
from utils.utils import extract_model, import_test_data
import pandas as pd
from models import linear_models

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', help='path to trainer model')
args = parser.parse_args()

model_path = Path(args.model_path)
assert(model_path.exists())

model_name = model_path.name.removesuffix(".pkl") 

trained_model = extract_model(model_path.absolute())

X_test = import_test_data('./data/test.csv')
if model_name in linear_models:
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
# else:
    # X_test = X_test.values

y_pred = trained_model.predict(X_test)

answerd_id = pd.read_csv('./data/test.csv')['PassengerId']

prediction = pd.DataFrame(y_pred, columns=["Survived"])
result_df = df = pd.concat([answerd_id, prediction], axis=1)
result_df.to_csv('submission.csv', index=False)