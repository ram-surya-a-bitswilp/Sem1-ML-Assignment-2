import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

_BREAST_CANCER_TRAIN_TEST_DATA = tuple()

class BreastCancerDataPreprocessor:
    """
    Preprocessor for Breast Cancer Wisconsin (Diagnostic) Dataset.
    Supports file paths, DataFrames, and dynamic batch scaling for standalone inference.
    """

    def __init__(
        self,
        target_column: str = 'diagnosis',
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.target_column = target_column
        self.drop_cols = drop_cols if drop_cols is not None else ['id']
        self.test_size = test_size
        self.random_state = random_state

        self.num_cols: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _encode_target(self, y: pd.Series) -> pd.Series:
        """Encodes 'M' (Malignant) as 1 and 'B' (Benign) as 0."""
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            mapping = {'M': 1, 'B': 0, 'm': 1, 'b': 0}
            return y.map(mapping).fillna(0).astype(int)
        return y.astype(int)

    def _build_pipeline(self) -> ColumnTransformer:
        """Creates numerical transformer pipeline for feature scaling."""
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        return ColumnTransformer(transformers=[
            ('num', num_pipeline, self.num_cols)
        ])

    def preprocess_from_dataframe(
        self,
        df: pd.DataFrame,
        is_training: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """
        Preprocesses an in-memory Pandas DataFrame directly.
        """
        df_clean = df.copy()
        df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]

        existing_drops = [col for col in self.drop_cols if col in df_clean.columns]
        df_clean = df_clean.drop(columns=existing_drops)

        if is_training:
            if self.target_column not in df_clean.columns:
                raise KeyError(f"Target column '{self.target_column}' missing during training.")

            X = df_clean.drop(columns=[self.target_column])
            y = self._encode_target(df_clean[self.target_column])

            self.num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            self.preprocessor = self._build_pipeline()
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            self.feature_names = self.num_cols
            self.is_fitted = True

            X_train_df = pd.DataFrame(X_train_processed, columns=self.feature_names, index=X_train.index).astype('float32')
            X_test_df = pd.DataFrame(X_test_processed, columns=self.feature_names, index=X_test.index).astype('float32')

            return X_train_df, X_test_df, y_train, y_test

        else:
            if self.target_column in df_clean.columns:
                X_val = df_clean.drop(columns=[self.target_column])
                y_val = self._encode_target(df_clean[self.target_column])
            else:
                X_val = df_clean
                y_val = None

            if not self.num_cols:
                self.num_cols = X_val.select_dtypes(include=['float64', 'int64']).columns.tolist()

            missing_cols = set(self.num_cols) - set(X_val.columns)
            if missing_cols:
                for col in missing_cols:
                    X_val[col] = np.nan

            if not self.is_fitted or self.preprocessor is None:
                self.preprocessor = self._build_pipeline()
                X_val_processed = self.preprocessor.fit_transform(X_val[self.num_cols])
                self.feature_names = self.num_cols
                self.is_fitted = True
            else:
                X_val_processed = self.preprocessor.transform(X_val[self.num_cols])

            X_val_df = pd.DataFrame(X_val_processed, columns=self.feature_names, index=X_val.index).astype('float32')

            return X_val_df, y_val

    def preprocess_from_csv(
        self,
        csv_path: str,
        is_training: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """Loads CSV and forwards to DataFrame processor."""
        df = pd.read_csv(csv_path)
        return self.preprocess_from_dataframe(df, is_training=is_training)


def breast_cancer_train_test_data():
    global _BREAST_CANCER_TRAIN_TEST_DATA

    if _BREAST_CANCER_TRAIN_TEST_DATA:
        return _BREAST_CANCER_TRAIN_TEST_DATA

    dp = BreastCancerDataPreprocessor()

    _BREAST_CANCER_TRAIN_TEST_DATA = dp.preprocess_from_csv("breast_cancer.csv", is_training=True)

    return _BREAST_CANCER_TRAIN_TEST_DATA