import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import time
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import StopCandidateException
import io
import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import xgboost as xgb

class GeminiAssistant:
    def __init__(self):
        # Configure Gemini API
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Store your API key in Streamlit secrets
        genai.configure(api_key=GOOGLE_API_KEY)

        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.chat = self.model.start_chat(history=[])

    def _send_message_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Helper method to send messages with retry logic and error handling"""
        for attempt in range(max_retries):
            try:
                response = self.chat.send_message(prompt)
                return response.text
            except StopCandidateException:
                # If we get a citation error, try with a modified prompt
                prompt = f"""
                Provide original analysis and suggestions based on this context.
                Do not quote or cite external sources. Focus on practical, specific advice:

                {prompt}
                """
                if attempt == max_retries - 1:
                    return "I encountered an error generating suggestions. Please try rephrasing your request."
                time.sleep(1)  # Short delay between retries
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"An error occurred: {str(e)}"
                time.sleep(1)

    def analyze_data_context(self, data_description: str) -> str:
        prompt = f"""
        Based solely on the following dataset information, provide original analysis:
        {data_description}

        Provide:
        1. Key observations about the data structure
        2. Specific analysis approaches tailored to this dataset
        3. Potential data quality concerns to address

        Keep suggestions specific to the data provided.
        """
        return self._send_message_with_retry(prompt)

    def explain_results(self, results_data: str) -> str:
        prompt = f"""
        Based solely on these specific results, provide original analysis:
        {results_data}

        Include:
        1. Direct interpretation of the metrics shown
        2. Practical implications for this specific case
        3. Concrete next steps based on these results
        """
        return self._send_message_with_retry(prompt)

    def get_code_suggestions(self, task_description: str) -> str:
        prompt = f"""
        Based solely on this specific dataset and task, suggest custom feature engineering steps:
        {task_description}

        Provide:
        1. Specific feature transformations for this dataset
        2. Custom feature creation ideas based on the available columns
        3. Data preprocessing steps tailored to these features

        Focus on practical implementation details without referencing external sources.
        """
        return self._send_message_with_retry(prompt)

class DataAnalyzer:
    def analyze_data(self, df: pd.DataFrame) -> dict:
        profile = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'correlations': df.corr().round(2).to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {}
        }
        return profile

class DataScienceAgent:
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.model_selector = EnhancedModelSelector()
        self.gemini_assistant = GeminiAssistant()

    def get_feature_suggestions(self, data: pd.DataFrame) -> str:
        """Wrapper method to get feature suggestions with enhanced context"""
        data_context = {
            'numeric_columns': data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': data.isnull().sum().to_dict(),
            'unique_values': {col: data[col].nunique() for col in data.columns}
        }

        task_description = f"""
        Dataset Properties:
        - Numeric features: {', '.join(data_context['numeric_columns'])}
        - Categorical features: {', '.join(data_context['categorical_columns'])}
        - Sample size: {len(data)} rows
        - Missing values: {sum(data_context['missing_values'].values())} total

        Suggest specific feature engineering steps for this dataset.
        """

        return self.gemini_assistant.get_code_suggestions(task_description)

    def process_data(self, df: pd.DataFrame, target_column: str):
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Fit model selector
        self.model_selector.fit(X, y)

        return self.model_selector

class AutoFeatureEngineer:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.target_encoder = None

    def _identify_features(self, df: pd.DataFrame) -> None:
        """Automatically identify feature types in the dataset"""
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                self.numeric_features.append(column)
            elif pd.api.types.is_datetime64_dtype(df[column]):
                self.datetime_features.append(column)
            else:
                self.categorical_features.append(column)

    def _engineer_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract useful features from datetime columns"""
        df_copy = df.copy()

        for col in self.datetime_features:
            try:
                df_copy[f'{col}_year'] = pd.to_datetime(df_copy[col]).dt.year
                df_copy[f'{col}_month'] = pd.to_datetime(df_copy[col]).dt.month
                df_copy[f'{col}_day'] = pd.to_datetime(df_copy[col]).dt.day
                df_copy[f'{col}_dayofweek'] = pd.to_datetime(df_copy[col]).dt.dayofweek
                df_copy[f'{col}_quarter'] = pd.to_datetime(df_copy[col]).dt.quarter

                # Drop original datetime column
                df_copy.drop(columns=[col], inplace=True)

                # Update numeric features list
                self.numeric_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day',
                    f'{col}_dayofweek', f'{col}_quarter'
                ])
            except Exception as e:
                print(f"Error processing datetime column {col}: {str(e)}")
                # If datetime conversion fails, treat as categorical
                if col in self.datetime_features:
                    self.datetime_features.remove(col)
                    if col not in self.categorical_features:
                        self.categorical_features.append(col)

        return df_copy

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        df_copy = df.copy()

        if len(self.numeric_features) >= 2:
            # Create interactions for most important numeric features (limit to avoid explosion)
            for i in range(len(self.numeric_features)):
                for j in range(i + 1, min(i + 3, len(self.numeric_features))):
                    try:
                        col1, col2 = self.numeric_features[i], self.numeric_features[j]
                        interaction_name = f'interaction_{col1}_{col2}'
                        df_copy[interaction_name] = df_copy[col1] * df_copy[col2]
                        self.numeric_features.append(interaction_name)
                    except Exception as e:
                        print(f"Error creating interaction feature: {str(e)}")
                        continue

        return df_copy

class AutoFeatureImplementer:
    def __init__(self):
        self.transformations = {}
        self.feature_history = []

    def _apply_numeric_transformations(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply common numeric transformations"""
        df_copy = df.copy()
        col_name = df_copy[column].name

        # Log transform for positive skewed data
        if df_copy[column].min() > 0:
            df_copy[f'{col_name}_log'] = np.log1p(df_copy[column])
            self.transformations[f'{col_name}_log'] = 'log transform'

        # Square root for positive skewed data
        if df_copy[column].min() >= 0:
            df_copy[f'{col_name}_sqrt'] = np.sqrt(df_copy[column])
            self.transformations[f'{col_name}_sqrt'] = 'square root'

        # Standard scaling
        df_copy[f'{col_name}_scaled'] = StandardScaler().fit_transform(df_copy[[column]])
        self.transformations[f'{col_name}_scaled'] = 'standard scaling'

        return df_copy

    def _create_polynomial_features(self, df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """Create polynomial features for numeric columns"""
        df_copy = df.copy()

        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    new_col = f'{col1}_{col2}_interaction'
                    df_copy[new_col] = df_copy[col1] * df_copy[col2]
                    self.transformations[new_col] = f'interaction between {col1} and {col2}'

        return df_copy

    def _bin_numeric_features(self, df: pd.DataFrame, column: str, n_bins: int = 5) -> pd.DataFrame:
        """Create binned versions of numeric features"""
        df_copy = df.copy()
        col_name = df_copy[column].name

        df_copy[f'{col_name}_binned'] = pd.qcut(df_copy[column], n_bins, labels=False, duplicates='drop')
        self.transformations[f'{col_name}_binned'] = f'binned into {n_bins} categories'

        return df_copy

    def implement_suggestions(self, df: pd.DataFrame, ai_suggestions: str) -> tuple[pd.DataFrame, dict]:
        """Implement feature engineering suggestions from AI"""
        df_result = df.copy()
        transformation_log = []

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Apply transformations based on data types
        for col in numeric_cols:
            # Numeric transformations
            df_result = self._apply_numeric_transformations(df_result, col)
            transformation_log.append(f"Applied numeric transformations to {col}")

            # Binning
            df_result = self._bin_numeric_features(df_result, col)
            transformation_log.append(f"Created bins for {col}")

        # Create polynomial features
        df_result = self._create_polynomial_features(df_result, numeric_cols)
        transformation_log.append("Created polynomial interaction features")

        # Handle categorical features
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                # One-hot encoding
                dummies = pd.get_dummies(df_result[col], prefix=col)
                df_result = pd.concat([df_result, dummies], axis=1)
                df_result.drop(columns=[col], inplace=True)
                transformation_log.append(f"One-hot encoded {col}")

        # Create summary of transformations
        summary = {
            'original_features': len(df.columns),
            'new_features': len(df_result.columns),
            'transformations': self.transformations,
            'transformation_log': transformation_log
        }

        return df_result, summary

class EnhancedModelSelector:
    def __init__(self):
        self.feature_engineer = AutoFeatureEngineer()
        self.best_model = None
        self.feature_pipeline = None
        self.problem_type = None
        self.target_encoder = None

    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline based on feature types"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.feature_engineer.numeric_features),
                ('cat', categorical_transformer, self.feature_engineer.categorical_features)
            ],
            remainder='drop'
        )

        return preprocessor

    def _get_model_candidates(self, problem_type: str) -> dict:
        """Get regularized model candidates with cross-validation"""
        if problem_type == 'classification':
            return {
                'rf': RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_samples=0.8,
                    max_features='sqrt',
                    random_state=42
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42,
                    use_label_encoder=False,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    gamma=1,
                    eval_metric='logloss'
                )
            }
        else:  # regression
            return {
                'rf': RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_samples=0.8,
                    max_features='sqrt',
                    random_state=42
                ),
                'xgb': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    gamma=1,
                    random_state=42
                ),
                'linear': Ridge(
                    alpha=10,
                    random_state=42
                )
            }

    def _determine_problem_type(self, y: pd.Series) -> str:
        """Determine if this is a classification or regression problem"""
        if pd.api.types.is_numeric_dtype(y):
            unique_values = len(np.unique(y))
            if unique_values < 10:  # Arbitrary threshold
                return 'classification'
            return 'regression'
        return 'classification'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the entire pipeline including feature engineering and model selection"""
        try:
            # Determine problem type
            self.problem_type = self._determine_problem_type(y)

            # Identify feature types
            self.feature_engineer._identify_features(X)

            # Engineer datetime features if any
            X_transformed = self.feature_engineer._engineer_datetime_features(X)

            # Create interaction features
            X_transformed = self.feature_engineer._create_interaction_features(X_transformed)

            # Encode target if classification
            if self.problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_transformed, y, test_size=0.2, random_state=42
            )

            # Store split data for later evaluation
            self.X_test = X_val
            self.y_test = y_val

            # Create preprocessing pipeline
            preprocessor = self._create_preprocessing_pipeline()

            # Get model candidates
            model_candidates = self._get_model_candidates(self.problem_type)

            # Train and evaluate models
            best_score = float('-inf')
            for name, model in model_candidates.items():
                try:
                    # Create pipeline
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('selector', SelectKBest(
                            f_classif if self.problem_type == 'classification' else f_regression,
                            k='all'
                        )),
                        ('model', model)
                    ])

                    # Fit pipeline
                    pipeline.fit(X_train, y_train)

                    # Evaluate
                    if self.problem_type == 'classification':
                        score = accuracy_score(y_val, pipeline.predict(X_val))
                    else:
                        score = r2_score(y_val, pipeline.predict(X_val))

                    if score > best_score:
                        best_score = score
                        self.best_model = pipeline

                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue

            if self.best_model is None:
                raise ValueError("No models were successfully trained")

            return self

        except Exception as e:
            raise Exception(f"Error in model fitting process: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet")

        # Apply same feature engineering steps
        X_transformed = self.feature_engineer._engineer_datetime_features(X)
        X_transformed = self.feature_engineer._create_interaction_features(X_transformed)

        return self.best_model.predict(X_transformed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions for classification problems"""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification problems")

        if self.best_model is None:
            raise ValueError("Model has not been fitted yet")

        # Apply same feature engineering steps
        X_transformed = self.feature_engineer._engineer_datetime_features(X)
        X_transformed = self.feature_engineer._create_interaction_features(X_transformed)

        return self.best_model.predict_proba(X_transformed)

