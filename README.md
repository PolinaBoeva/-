# Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API

## Описание задания:
В рамках проекта был проведен анализ исторических данных температуры для выявления сезонных закономерностей и аномалий, а также был интегрирован API OpenWeatherMap для мониторинга текущей температуры в различных городах. 

## Результаты анализа:

### 1. Анализ временных рядов

- **Вычисление скользящего среднего и стандартного отклонения**: Для сглаживания колебаний температур было вычислено 30-дневное скользящее среднее, которое помогает отфильтровывать краткосрочные колебания и выявлять долгосрочные тренды. Также было рассчитано стандартное отклонение для оценки степени изменчивости температуры.
  
- **Определение аномалий**: На основе данных о скользящем среднем и стандартном отклонении были выявлены аномалии. Все температуры, которые отклонялись более чем на 2 стандартных отклонения от скользящего среднего, были классифицированы как аномалии. Например, резкие повышения или падения температуры, выходящие за пределы нормы для конкретного времени года, были отмечены как потенциальные аномалии.

- **Построение долгосрочных трендов изменения температуры**: Для каждого города на основе коэффициента наклона линейной регрессии были сделаны выводы о положительном/отрицательном долгосрочном тренде.

- **Сезонные профили с указанием среднего и стандартного отклонения**: На основе сезонных данных температур были построены графики, отражающие сезонные профили.

### 2. Мониторинг текущей температуры
Для мониторинга текущей температуры использовалось **OpenWeatherMap API**, что позволило получить актуальные данные по температуре для разных городов. 

- **Сравнение с историческим диапазоном**: Текущая температура в разных городах была сравнена с историческими данными для текущего сезона.

### 3. Разработка интерактивного приложения
Было разработано интерактивное приложение на Streamlit: <https://temperatureanalysisandmonitoring.streamlit.app>

- **Выбирать города** для анализа температурных данных.
- **Просматривать результаты анализа**, включая графики температурных рядов, сезонные профили и аномалии.
- **Сравнивать текущую температуру** с историческими данными для выбранного города.

Интерфейс приложения интуитивно понятен и позволяет пользователю легко анализировать данные о температуре и получать информацию о текущем климате в реальном времени.

## Эксперименты и оптимизация:

### 1. Распараллеливание анализа
Для повышения производительности был проведен эксперимент с распараллеливанием анализа. 

- **Распараллеливание анализа**: Была использована библиотека **ProcessPoolExecutor** для параллельной обработки данных для каждого города, значительного ускорения относительно обычного расчета на pandas выявлено не было.  

### 2. Синхронные и асинхронные методы для получения текущей температуры
Для получения текущей температуры через OpenWeatherMap API мы протестировали два подхода: синхронный и асинхронный.

- **Синхронный метод**: В синхронном режиме отправлялись запросы к API последовательно для каждого города. Этот подход работает корректно, но может занимать длительное время при необходимости запросить данные для множества городов.

- **Асинхронный метод**: Используя библиотеку **`aiohttp`**, был реализован асинхронный подход, при котором можно отправлять несколько запросов  одновременно, не блокируя выполнение программы. 

- **Результаты**: При единичных запросах значительной разницы во времени выполнения выявлено не было, при этом асинхронный метод будет быстрее при запросах данных о нескольких городах.


## Технологический стек:

- **Язык программирования**: Python
- **Библиотеки для анализа данных и экспериментов**: Pandas, NumPy, Matplotlib, Seaborn, ProcessPoolExecutor, aiohttp, requests
- **API для получения данных**: OpenWeatherMap API
- **Веб-технологии для приложения**: Streamlit

