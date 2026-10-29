import numpy as np
import pandas as pd

NOMINAL = ['Marital Status', 'Application mode', 'Course', 'Previous qualification', 'Nacionality',
           "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]

NEW_A = ['mature_student', 'grade_gain']
NEW_B = ['approved_ratio_1', 'pass_rate_eval_1', 'zero_approved_1', 'payment_problem', 'grade_x_ratio_1']


def add_features(X):
    """Новые признаки; группа считается, только если есть исходные столбцы."""
    X = X.copy()
    X['mature_student'] = (X['Age at enrollment'] >= 25).astype(int)
    X['grade_gain'] = X['Admission grade'] - X['Previous qualification (grade)']
    if 'Curricular units 1st sem (enrolled)' in X:
        enr = X['Curricular units 1st sem (enrolled)'].replace(0, np.nan)
        ev = X['Curricular units 1st sem (evaluations)'].replace(0, np.nan)
        app = X['Curricular units 1st sem (approved)']
        X['approved_ratio_1'] = app / enr                 # доля сданных дисциплин
        X['pass_rate_eval_1'] = app / ev                  # доля успешных аттестаций
        X['zero_approved_1'] = ((app == 0) & (X['Curricular units 1st sem (enrolled)'] > 0)).astype(int)
        X['grade_x_ratio_1'] = X['Curricular units 1st sem (grade)'] * X['approved_ratio_1'].fillna(0)
    if 'Debtor' in X:
        X['payment_problem'] = ((X['Debtor'] == 1) | (X['Tuition fees up to date'] == 0)).astype(int)
    return X


def to_category(X):
    X = X.copy()
    cols = [c for c in NOMINAL if c in X]
    X[cols] = X[cols].astype('category')
    return X


def prepare(X, cols):
    """add_features -> выбор столбцов -> категориальные типы."""
    return to_category(add_features(X)[cols])
