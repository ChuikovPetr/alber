# alber

- [Установка](#installing)
- [Общая концепция модели и результаты](#model)
- [Общее устройство проекта](#project)
- [Использование - Fitting](#fitting)
- [Использование - Forecasting](#forecasting)
- [Features](#features)
- [Features for the best model (detail)](#features_detail)
- [Train/Test splitting](#splitting_train_test)
- [Walk-Forward validation](#walk_forward)
- [One factor analysis](#one_factor)
- [Multifactor analysis](#multifactor)
- [Fine tuning](#fine_tuning)
- [Possible improvments](#improvements)



# Установка <a name="installing"></a>
Перед использованием проекта необходимо его установить с помощью следующих команд:

```bash
git clone git@github.com:ChuikovPetr/alber.git
cd alber/
pip install -r requirements.txt
pip install .
```


# Общая концепция модели и результаты <a name="model"></a>

Для моделирования использовалась модель градиентного бустинга из библиотеки LightGBM.
Про используемые признаки подробнее читай ниже.

Далее приведены результаты 3 моделей на out-of-time. Само собой нельзя сравнивать модели на тесте и после этого выбирать лучшую, так как мы просто подстроимся под тест. Но, наверняка, специалисты из Alber Blanc будут проверять модель на своем out-of-time. Тогда можем считать мою проверку на out-of-time предтестом.

Далее результаты 3 моделей:
1. model_mf_15f - лучшая модель, полученная после FS.
2. baseline_1f - бэйзлайн на основе 1 признака, показавшего наилучшие результаты при одномерном анализе.
3. model_of_29f - бэйзлайн на основе 29 признаков, полученных после процедуры однофакторного анализа.

| Название модели  | Метрика R^2 на out-of-time (10% от всего сэмпла) |
| ------------- | ------------- |
| model_mf_15f  | 0.0193  |
| Baseline установленный Alber Blanc в ТЗ | 0.01  |
| baseline_1f  | 0.007  |
| model_of_29f  | 0.004  |

# Общее устройство проекта <a name="project"></a>

Решение организовано в виде предустанавливаемого проекта python.
Структура:
1. Папка folder с необходимыми модулями
\n\t1.1. load_data.py - содержит функции для загрузки данных из h5 формата в Pandas DataFrame.
\n\t1.2. feature_generation.py - содержит функции для генерации признаков.
\n\t1.3. wf_splitting_data.py - содержит функции для разбиения данных (Walk-Forward и взятие обычного сэмпла).
\n\t1.4. train.py - содержит функции для разбиения данных (Walk-Forward и взятие обычного сэмпла).

# Использование - Fitting <a name="fitting"></a>
Результатом работы являются 2 скрипта (как указано в тестовом задании):
1. notebooks/Result - Fitting.ipynb
2. notebooks/Result - Forecasting.ipynb

Первый скрипт Fitting принимает в качестве параметра имя папки с файлами для обучения модели (data.h5 и result.h5), затем рассчитывает признаки на основе этих данных, после чего проводит обучение моделей и валидацию на out-of-time данных.

Важно! В файле обучается 3 модели: baseline_1f, model_of_29f, model_mf_15f, если вы хотите обучить только одну просто не запускайте соответствующую клетку.

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


# Features for the best model (detail) <a name="features_detail"></a>

1. stoch_k_price_21_1 - быстрая линия (без сглаживания, то есть параметры стохастика [21, 1, _]) стохастического осциллятора, рассчитанная на основе цены.
2. wap_balance = wap1 / wap2, где wapi рассчитывается для i-ого bid/ask соответственно.
3. volume_imbalance = (ask_size1 + ask_size2) - (bid_size1 + bid_size2)
4. rel_order_count_1_80 - отношение количества ордеров для трейдов за последнюю единицу времени к своей скользящей средней за 80 единиц времени.
5. rel_price_5_10 - отношение скользящих средних для цены за периоды 5 и 10 соответственно.
6. log_return_mean_price = log((bid_price1 + ask_price1) / 2)
7. bid_spread = bid_price1 / bid_price2 - 1
8. ask_spread = ask_price1 / ask_price2 - 1
9. Money = price * size - количество торгуемых денег за текущую единицу времени.
10. rel_stoch_price_21_1_3 - отношение быстрой линии стохастического осциллятора, рассчитанного для цены, к медленной. Параметры стохастического осциллятора: [21, 1, 3].
11. rel_volume_ask_bid1 = ask_size1 / bid_size1 - 1
12. bid_ask_spread2 = ask_price2 / bid_price2 - 1
13. total_volume = (ask_size1 + ask_size2) + (bid_size1 + bid_size2)
14. rel_price_1_5 - отношение цены к своей скользящей средней с периодом 5.
15. rel_price_40_80  - отношение скользящих средних для цены за периоды 40 и 80 соответственно.

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
2. Отсек сильноскоррелированные признаки (оставив наиболее сильный из 2 сколлирированных).

# Multifactor analysis <a name="multifactor"></a>

Чтобы подобрать стабильный самый сильный набор признаков использовался алгоритм последовательного добавления признаков в модель.
Поскольку данные достаточно шумные, необходимо было использовать стат. тест для связных выборок поверх данных метрики для оценки производительности каждого нового множества признаков (признак добавлялся в модель, если p_value < 0.05).

# Fine tuning <a name="fine_tuning"></a>

Данные оказались слишком шумные для точной настройки параметров модели - не получилось стат. значимо увеличить метрику, изменяя параметры lightgbm. В итоге оставил feature_fraction_by_node == 0.6 (использовал как дефолтное значение), остальные параметры не менял.


# Possible improvments <a name="improvements"></a>

Если бы было больше времени, можно было бы попробовать следующее:

1. Использовать WAP цену из биржевого стакана для генерации признаков, аналогичных признакам для Executed trades.
2. Использовать все 30 цен/объемов для генерации фичей для биржевого стакана (я использовал 2 лучших для Bid/Ask соответственно).
3. Попробовать другие модели: реккурентные нейронные сети и так далее.
4. Попробовать другие подходы к моделированию. Например, можно было сначала понизить размерность признакового пространства автокодировщиком и только потом применять модель. Или применять сначала шумоподавляющий автокодировщик.
5. Придумать лучшие фичи. Например, добавить других функций из технического анализа (MACD, CO, RSI, Aroon, ...).
