# Отчет по проекту классификации SMS-спама

## 1. Описание проекта
Проект направлен на создание модели машинного обучения для классификации SMS-сообщений на спам и не-спам (ham). 

### Использованные технологии:
- Python 3
- pandas (обработка данных)
- scikit-learn (машинное обучение)
- NLTK (обработка текста)
- NumPy (численные вычисления)

## 2. Реализация

### 2.1 Предобработка данных
1. Загрузка данных из файла SMSSpamCollection
2. Предобработка текста:
   - приведение к нижнему регистру
   - токенизация с сохранением важных символов
   - лемматизация
   - удаление стоп-слов

### 2.2 Извлечение признаков
1. Текстовые признаки:
   - TF-IDF векторизация
   - использование униграмм, биграмм и триграмм
   - ограничение максимального количества признаков
2. Дополнительные признаки:
   - длина сообщения
   - количество слов
   - средняя длина слова
   - доля заглавных букв
   - количество цифр
   - количество восклицательных и вопросительных знаков
   - количество символов $ и %
   - количество URL-адресов
   - количество email-адресов
   - количество телефонных номеров
   - доля специальных символов

### 2.3 Модель
Использован ансамбль из трех моделей:
1. MultinomialNB (работает только с текстовыми данными)
2. RandomForest (использует все признаки)
3. LinearSVC (использует все признаки)

Финальное решение принимается голосованием большинства.

## 3. Результаты

### 3.1 Метрики качества
- Общая точность (accuracy): 99%

Для не-спам сообщений:
- Precision: 99%
- Recall: 100%
- F1-score: 99%

Для спам сообщений:
- Precision: 99%
- Recall: 92%
- F1-score: 95%

### 3.2 Матрица ошибок
- Правильно определенные не-спам сообщения: 965
- Ложные срабатывания (не-спам принят за спам): 1
- Пропущенные спамы: 12
- Правильно определенные спамы: 137

## 4. Анализ результатов

### 4.1 Сильные стороны
1. Очень высокая общая точность (99%)
2. Минимальное количество ложных срабатываний (всего 1)
3. Высокий процент обнаружения спама (92%)
4. Сбалансированные показатели для обоих классов
5. Надежное определение сложных случаев

### 4.2 Слабые стороны
1. Небольшой процент пропущенного спама (8%)
2. Сложность модели (требуется больше вычислительных ресурсов)

## 5. Возможные улучшения

### 5.1 Технические улучшения:
1. Тонкая настройка гиперпараметров моделей
2. Добавление других моделей в ансамбль
3. Экспериментирование с весами голосов разных моделей
4. Расширение набора признаков

### 5.2 Практические улучшения:
1. Добавление механизма периодического переобучения
2. Создание API для интеграции
3. Добавление объяснения решений модели
4. Оптимизация производительности

## 6. Заключение
Разработанная модель показывает отличные результаты в классификации SMS-сообщений. Использование ансамбля моделей и расширенного набора признаков позволило достичь высокой точности при минимальном количестве ошибок. Модель успешно справляется как с очевидными, так и с более сложными случаями спама, что делает её пригодной для практического применения. 