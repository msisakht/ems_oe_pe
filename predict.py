import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_labels = tf.argmax(y_pred, axis=1)
        y_true_labels = tf.argmax(y_true, axis=1)
        self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
        self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


def load_model(model_type, model_format="tf"):
    if model_format == "tf":
        model_path = f"models/tf_model_{model_type}.keras"
        with keras.utils.custom_object_scope({'F1Score': F1Score}):
            model = keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            metrics=[F1Score()])
    elif model_format == "xgb":
        model_path = f"models/xgb_model_{model_type}.model"
        model = load(model_path)
    else:
        raise ValueError("Unsupported model format. Use 'tf' for TensorFlow or 'xgb' for XGBoost.")
    return model


def get_valid_syms(model_type, model_format):
    valid_genes_path = f"data/{model_format}_valid_genes_{model_type}.pkl"
    with open(valid_genes_path, "rb") as file:
        data = pickle.load(file)
    return data["valid_genes"]


def rm_dups(df):
    numeric_cols = df.select_dtypes(include='number').columns
    def row_iqr(row):
        q1 = row.quantile(0.25)
        q3 = row.quantile(0.75)
        return q3 - q1
    df['Row_IQR'] = df[numeric_cols].apply(row_iqr, axis=1)
    df_sorted = df.sort_values(by=['Symbols', 'Row_IQR'], ascending=[True, False])
    df_deduped = df_sorted.drop_duplicates(subset='Symbols', keep='first')
    df_final = df_deduped.drop(columns=['Row_IQR'])
    return df_final


def clean_syms(df):
    df['Symbols'] = df['Symbols'].apply(lambda x: re.split(r'[|/,;]', x)[0] if isinstance(x, str) else x)
    return df


def preprocess(data_path, valid_genes):
    df = pd.read_csv(data_path)
    if "Symbols" not in df.columns:
        raise ValueError("The input CSV file must contain a 'Symbols' column with gene names.")
    df = clean_syms(df)
    df = rm_dups(df)
    df.set_index("Symbols", inplace=True)
    missing_genes = set(valid_genes) - set(df.index)
    if missing_genes:
        print("Warning: The following genes are missing, which may affect predictions:", missing_genes)
    df = df.reindex(valid_genes, fill_value=0.0)
    return df.T


def predict(model, processed_data, model_format):
    if model_format == "tf":
        predictions = model.predict(processed_data)
        return np.argmax(predictions, axis=1)  
    elif model_format == "xgb":
        predictions = model.predict(processed_data)
        return predictions
    else:
        raise ValueError("Unsupported model format. Use 'tf' for TensorFlow or 'xgb' for XGBoost.")


def main(input_data_path, model_type, model_format):
    print(f"Loading {model_type} model ({model_format})...")
    model = load_model(model_type, model_format)
    print(f"Getting valid gene symbols for {model_type} ({model_format})...")
    valid_genes = get_valid_syms(model_type, model_format)
    print("Preprocessing input data...")
    processed_data = preprocess(input_data_path, valid_genes)
    print("Making predictions...")
    results = predict(model, processed_data, model_format)
    print("Predictions:")
    for sample, pred in zip(processed_data.index, results):
        print(f"{sample}: {pred}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run models on gene expression data.")
    parser.add_argument("-d", "--data", required=True, help="Path to input data CSV file")
    parser.add_argument("-m", "--model", choices=["tf", "xgb"], required=True, help="Model format (tf or xgb)")
    parser.add_argument("-t", "--type", choices=["PE", "OE"], required=True, help="Model type (PE or OE)")
    args = parser.parse_args()
    main(args.data, args.type, args.model)
