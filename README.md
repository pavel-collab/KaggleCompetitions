```
mkdir data && cd data
kaggle competitions download -c titanic
unzip titanic.zip
```

```
python3 ./src/train.py
python3 ./src/eval.py
python3 ./src/predict.py -m ./saved_models/svm.py
```

### TODO:

- попробовать KFold, поскольку данных не так много
- универсальный feature generator, генерация и отбор фичей по их значимости