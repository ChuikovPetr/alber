# alber

- [Установка](#installing)
- [Использование - Fitting](#fitting)
- [Использование - Forecasting](#forecasting)
- [Features](#features)
- [Train/Test splitting](#splitting_train_test)
- [Walk-Forward validation](#walk_forward)
- [One factor analysis](#one_factor)
- [Multifactor analysis](#multifactor)
- [Fine tuning](#fine_tuning)
- [Possible improvment](#improvements)



# Установка <a name="installing"></a>
Перед использованием проекта необходимо его установить с помощью следующих команд:

```bash
git clone git@github.com:ChuikovPetr/alber.git
cd alber/
pip install -r requirements.txt
pip install .
```


# Использование - Fitting <a name="fitting"></a>
Результатом работы являются 2 скрипта (как указано в тестовом задании):
1. Result - Fitting.ipynb
2. Result - Forecasting.ipynb

Первый скрипт Fitting принимает в качестве параметра имя папки с файлами для обучения модели (data.h5 и result.h5), затем рассчитывает признаки на основе этих данных, после чего проводит обучение моделей и валидацию на out-of-time данных (в файле обучается 3 модели baseline_1f, model_of_29f, model_mf_8f)

Общие параметры
```python
base = Path('../../Storage/alber') # Имя папки с файлами для обучения модели

train_val_ratio = 0.1 # Сколько данных для
test_size = 985_564 # Сколько единиц времени для out-of-time
```

Чтобы обучить конкретную модель нужно задать 2 параметра: имя модели и ключ в словаре dict_features, чтобы подгрузить соответствующие фичи. Далее пример:

```python
features_mode = 29
name_exp = 'model_of_29f'

r2_score_test_29f = train(
    train_vitrine[['time', 'target'] + dict_features[features_mode]], # train_vitrine рассчитывается ранее в том же блокноте
    name_exp,
    train_val_ratio,
    test_size
)
```
