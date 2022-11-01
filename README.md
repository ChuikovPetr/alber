# alber

- [Установка](#installing)
- [Использование - Fitting](#fitting)
- [Использование - Forecasting](#forecasting)
- [Общая концепция модели](#model)
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
1. notebooks/Result - Fitting.ipynb
2. notebooks/Result - Forecasting.ipynb

Первый скрипт Fitting принимает в качестве параметра имя папки с файлами для обучения модели (data.h5 и result.h5), затем рассчитывает признаки на основе этих данных, после чего проводит обучение моделей и валидацию на out-of-time данных.

Важно! В файле обучается 3 модели: baseline_1f, model_of_29f, model_mf_8f, если вы хотите обучить только одну просто не запускайте соответствующую клетку.

Общие параметры:

```python
base = Path('../../Storage/alber') # Имя папки с файлами для обучения модели

train_val_ratio = 0.1 # Сколько данных для train_val - используется для предостанова
test_size = 985_564 # Сколько единиц времени для out-of-time
```

Чтобы обучить конкретную модель нужно задать 2 параметра: имя модели и ключ в словаре dict_features, чтобы подгрузить соответствующие признаки:

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

# Использование - Forecasting <a name="forecasting"></a>


# Общая концепция модели <a name="model"></a>

Для моделирования использовалась модель градиентного бустинга из библиотеки LightGBM.
Про используемые признаки подробнее читай ниже.

# Features <a name="features"></a>

Функции относящиеся к генерации фичей находятся в модуле alber/feature_generation.py
Общее время генерации всех признаков для предоставленного датасета (почти 10_000_000 строк) составило около 32 часов на MacBook Pro.
Для построения модели у нас есть 2 типа данных:

1. Order book snapshots - данные по биржевому стакану.
2. Executed trades - данные по фактической торговле.

# Order book snapshots features

Имеем по 30 лучших предложений для Bid/Ask соответственно. Для экономии вычислительных ресурсов будем генировать признаки только на первых 2 самых лучших предложениях Bid/Ask соответственно.

Функция верхнего уровня для генерации фичей для биржевого стакана book_preprocessor.

# Executed trades features

Признаки, сгенерированные на данных по реальной торговле можно разделить на 3 основных типа:

1. Скользящие средние (функция get_features_ma). Используемые поля: price, size, order_count, Money.
2. Стохастические осцилляторы (функция get_features_stoch). Используемые поля: price, size.
3. Максимальной отклонение на основе z_score (функция get_features_zscore). Используемые поля (не базовые): Ret, Sprd, Sprd_Up, Sprd_Down.

# Train/Test splitting  <a name="splitting_train_test"></a>

В ноутбуке notebooks/1. BUILD - Read data.ipynb была выделена тестовая выборка (out-of-time) для тестирования итоговой модели.
Функция для разделения create_oot лежит в модуле alber/wf_splitting_data

Под тест было выделено 985_564 наблюдений (около 10% всего датасета).

# Walk-Forward validation <a name="walk_forward"></a>

Для выбора признаков для модели и настроек параметров использовалась Walk-Forward валидация.
После исключения тестовых данных из рассмотрения, оставшиеся данные были разбиты на 10 фолдов по принципу Walk-Forward, при этом для каждого фолда у нас было 3 множества данных:

1. train_i (непосредственно для обучения).
2. train_val_i (для предостановки модели, чтобы избежать переобучения).
3. val_i (для валидации).

# One factor analysis <a name="one_factor"></a>

Чтобы уменьшить множество рассматриваемых фичей для тяжелых алгоритмов Multifactor analysis, необходимо провести One factor analysis.
Для этого я провалидировал модель на каждом признаке на наших 10 Walk-Forward фолдах. При этом модель обучалась по 5 раз на каждом фолде (чтобы уменьшить влияние удачного seed). Итого, имеем по 10*5=50 данных метрики для оценки производительности каждого признака.

После я провел следующие действия:
1. Отсек самые слабые признаки.
2. Отсек сильноскоррелированные признаки (оставив из 2 сколлирированных - наиболее сильный).

# Multifactor analysis <a name="multifactor"></a>

Чтобы подобрать стабильный самый сильный набор признаков использовался алгоритм последовательного добавления признаков в модель.
Поскольку данные достаточно шумные, необходимо было использовать стат. тест для связных выборок поверх данных метрики для оценки производительности каждого нового множества признаков (признак добавлялся в модель, если p_value < 0.05).

# Fine tuning <a name="fine_tuning"></a>

Данные оказались слишком шумные для точной настройки параметров модели - не получилось стат. значимо увеличить метрику, изменяя параметры lightgbm. В итоге оставил feature_fraction_by_node == 0.6 (использовал как дефолтное значение), остальные параметры не менял.


# Possible improvment <a name="improvements"></a>

Если бы было больше времени, можно было бы попробовать следующее:

1. Использовать WAP цену из биржевого стакана для генерации признаков, аналогичных признакам для Executed trades.
2. Использовать все 30 цен/объемов для генерации фичей для биржевого стакана (я использовал 2 лучших для Bid/Ask соответственно).
3. Попробовать другие модели: реккурентные нейронные сети и так далее.
4. Попробовать другие подходы к моделированию. Например, можно было сначала понизить размерность признакового пространства автокодировщиком и только потом применять модель. Или применять сначала шумоподавляющий автокодировщик.
5. Придумать лучшие фичи. Например, добавить других функций из технического анализа (MACD, CO, RSI, Aroon, ...).
