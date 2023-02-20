import json

import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import numpy as np


def load_data(file_name):
    reviews = []
    ratings = []
    with open(file_name, 'r') as data:
        data_review = data.readlines()
        for line in data_review:
            review_dictionary = json.loads(line)
            if 'reviewText' in review_dictionary.keys():
                reviews.append(review_dictionary['reviewText'])
                ratings.append(review_dictionary['overall'])
                if 'summary' in review_dictionary.keys():
                    reviews[-1] += '\n' + review_dictionary['summary']
    return reviews, np.array(ratings).astype(int)


def print_k_best(data, train_ratings, vocabulary):
    k_best = SelectKBest(k=15)
    k_best.fit(data, train_ratings)
    k_best_results = k_best.get_support(indices=False)
    words = []
    print("Best 15 words:")
    for indx, isBest in enumerate(k_best_results):
        if isBest:
            words.append(vocabulary[indx][0])
    print(words)


def classify(train_file, test_file):
    # Load Train Files
    train_reviews, train_ratings = load_data(train_file)
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # Initialize Transform and Classifiera
    transformed_text = Pipeline(
        [('vect', CountVectorizer(ngram_range=(1, 1), max_features=1000)), ('tfidf', TfidfTransformer())])
    text_clf = Pipeline([('transformed', transformed_text), ('clf', LogisticRegression(random_state=0, max_iter=200))])

    # Fit Model to Pipline
    text_clf.fit(train_reviews, train_ratings)

    voc = sorted(transformed_text['vect'].vocabulary_.items(), key=lambda x: x[1])

    # Load Test Files, and test the Model
    test_reviews, test_ratings = load_data(test_file)
    predicts = text_clf.predict(test_reviews)
    accuracy = numpy.mean(predicts == test_ratings)
    f1 = f1_score(test_ratings, predicts, average=None)

    print_k_best(transformed_text.transform(train_reviews), train_ratings, voc)

    # Print Confusion Matrix
    confus_matrix = confusion_matrix(test_ratings, predicts, labels=text_clf.classes_)
    print("\nConfusion Matrix:")
    print(confus_matrix, '\n')

    test_results = {'class_1_F1': f1[0],
                    'class_2_F1': f1[1],
                    'class_3_F1': f1[2],
                    'class_4_F1': f1[3],
                    'class_5_F1': f1[4],
                    'accuracy': accuracy}

    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
