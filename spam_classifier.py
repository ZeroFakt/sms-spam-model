import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_data():
    try:
        # Отключаем проверку SSL для загрузки данных NLTK
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Загружаем необходимые ресурсы NLTK
        resources = ['punkt_tab', 'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            nltk.download(resource)
        print("NLTK данные успешно загружены")
    except Exception as e:
        print(f"Ошибка при загрузке NLTK данных: {e}")

def extract_features(text):
    """Расширенное извлечение признаков из текста"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'capitals_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'has_numbers': any(c.isdigit() for c in text),
        'numbers_count': sum(c.isdigit() for c in text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'dollar_count': text.count('$'),
        'percentage_count': text.count('%'),
        'urls_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'email_count': len(re.findall(r'[\w\.-]+@[\w\.-]+', text)),
        'phone_count': len(re.findall(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10}', text)),
        'special_chars_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / len(text) if len(text) > 0 else 0,
    }
    return features

def preprocess_text(text, lemmatizer):
    # Сохраняем оригинальные признаки
    features = extract_features(text)
    
    # Базовая предобработка текста
    text = text.lower()
    
    # Токенизация с сохранением важных символов
    text = re.sub(r'([^\w\s!?$%])|([^\w\s])', r' \1 ', text)
    tokens = word_tokenize(text)
    
    # Лемматизация и удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    processed_text = ' '.join(tokens)
    return processed_text, features

def prepare_data(df):
    lemmatizer = WordNetLemmatizer()
    
    # Предобработка текстовых данных и извлечение признаков
    processed_data = [preprocess_text(msg, lemmatizer) for msg in df['message']]
    df['processed_message'] = [data[0] for data in processed_data]
    
    # Добавление дополнительных признаков
    features_df = pd.DataFrame([data[1] for data in processed_data])
    for column in features_df.columns:
        df[f'feature_{column}'] = features_df[column]
    
    # Нормализация числовых признаков
    numeric_features = [col for col in df.columns if col.startswith('feature_')]
    df[numeric_features] = (df[numeric_features] - df[numeric_features].mean()) / df[numeric_features].std()
    
    # Преобразование меток
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df

def create_ensemble_model():
    # Создаем базовые модели
    # Используем только TF-IDF признаки для MultinomialNB
    nb = Pipeline([
        ('vectorizer', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2
        )),
        ('classifier', MultinomialNB(alpha=0.1))
    ])

    # Используем все признаки для RF и SVM
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )
    
    svm = LinearSVC(
        class_weight='balanced',
        random_state=42,
        max_iter=10000
    )
    
    return [nb, rf, svm]

def main():
    # Загрузка данных и NLTK ресурсов
    download_nltk_data()
    df = pd.read_csv('sms-spam-collection/SMSSpamCollection', encoding='latin-1', sep='\t', names=['label', 'message'])
    
    # Предобработка данных
    df = prepare_data(df)
    
    # Разделение на признаки и целевую переменную
    X_text = df['processed_message']
    X_features = df[[col for col in df.columns if col.startswith('feature_')]]
    y = df['label']
    
    # Разделение на обучающую и тестовую выборки
    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Векторизация текста для RF и SVM
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=2,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    X_text_train_vectorized = vectorizer.fit_transform(X_text_train)
    X_text_test_vectorized = vectorizer.transform(X_text_test)
    
    # Объединение признаков для RF и SVM
    X_train_combined = np.hstack([X_text_train_vectorized.toarray(), X_features_train])
    X_test_combined = np.hstack([X_text_test_vectorized.toarray(), X_features_test])
    
    # Обучение моделей
    models = create_ensemble_model()
    
    # Обучаем NB на текстовых данных
    models[0].fit(X_text_train, y_train)
    
    # Обучаем RF и SVM на комбинированных данных
    models[1].fit(X_train_combined, y_train)
    models[2].fit(X_train_combined, y_train)
    
    # Получаем предсказания от каждой модели
    nb_pred = models[0].predict(X_text_test)
    rf_pred = models[1].predict(X_test_combined)
    svm_pred = models[2].predict(X_test_combined)
    
    # Голосование большинством
    predictions = np.array([nb_pred, rf_pred, svm_pred])
    final_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=predictions
    )
    
    # Вывод результатов
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_predictions))
    
    # Тестовые примеры
    test_messages = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
        "Hey, what time will you be home for dinner?",
        "URGENT! Your Mobile No. was awarded £2000 Bonus Caller Prize",
        "Call me when you get a chance, need to discuss the project",
    ]
    
    # Предобработка тестовых сообщений
    lemmatizer = WordNetLemmatizer()
    test_processed = [preprocess_text(msg, lemmatizer) for msg in test_messages]
    test_text = [data[0] for data in test_processed]
    test_features = pd.DataFrame([data[1] for data in test_processed])
    
    # Получение предсказаний для тестовых сообщений
    nb_test_pred = models[0].predict(test_text)
    
    test_text_vectorized = vectorizer.transform(test_text)
    test_combined = np.hstack([test_text_vectorized.toarray(), test_features])
    rf_test_pred = models[1].predict(test_combined)
    svm_test_pred = models[2].predict(test_combined)
    
    # Финальное голосование
    test_predictions = np.array([nb_test_pred, rf_test_pred, svm_test_pred])
    final_test_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 
        axis=0, 
        arr=test_predictions
    )
    
    print("\nТестовые предсказания:")
    for message, prediction in zip(test_messages, final_test_predictions):
        print(f"Сообщение: {message}")
        print(f"Предсказание: {'СПАМ' if prediction == 1 else 'НЕ СПАМ'}\n")

if __name__ == "__main__":
    main() 