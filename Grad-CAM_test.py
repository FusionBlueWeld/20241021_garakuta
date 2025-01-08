import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os
import argparse
import matplotlib.pyplot as plt

# データファイルのパス
DATA_FILE = 'data/synthetic_waveforms.h5'
MODEL_SAVE_PATH = 'model/best_model.keras'
PREDICTION_CSV_PATH = 'predictions.csv'
GRAD_CAM_OUTPUT_DIR = 'grad_cam_output'

def load_data(file_path):
    """HDF5ファイルから波形データとパラメータをロードする関数"""
    with h5py.File(file_path, 'r') as f:
        y_data = f['y'][:]
        parameter_data = f['parameters'][:]
    return y_data, parameter_data

def create_model(input_shape, num_params):
    """波形データからパラメータを予測するCNNモデルを作成する関数"""
    input_layer = Input(shape=input_shape)
    cnn1 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv1d_1')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', name='conv1d_2')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(cnn2)
    cnn3 = Conv1D(filters=256, kernel_size=5, activation='relu', padding='same', name='conv1d_3')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(cnn3)
    flatten = Flatten()(pool3)
    dense = Dense(128, activation='relu')(flatten)
    output_layer = Dense(num_params)(dense)  # パラメータ数を出力層の次元数とする

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def train_model(model, x_train, y_train, x_val, y_val, save_path):
    """モデルを訓練する関数"""
    model.compile(optimizer='adam', loss='mse')  # 回帰問題なので損失関数はMSE
    checkpoint = ModelCheckpoint(filepath=save_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=50,  # エポック数は調整してください
                        batch_size=32,
                        callbacks=[checkpoint])
    return history

def evaluate_model(model_path, x_test, y_test, csv_path):
    """保存されたモデルをロードし、テストデータで評価する関数"""
    best_model = tf.keras.models.load_model(model_path)
    predictions = best_model.predict(x_test)

    mse_scores = mean_squared_error(y_test, predictions, multioutput='raw_values')
    r2_scores = r2_score(y_test, predictions, multioutput='raw_values')

    parameter_names = ["mu", "sigma", "offset", "amplitude", "exp_decay_rate", "exp_amplitude_factor"]
    print("評価指標:")
    for i, name in enumerate(parameter_names):
        print(f"  {name} - MSE: {mse_scores[i]:.4f}, R^2: {r2_scores[i]:.4f}")

    # CSVファイルに結果を保存
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = []
        for name in parameter_names:
            header.extend([f"{name}_true", f"{name}_predicted"])
        writer.writerow(header)

        for true_params, predicted_params in zip(y_test, predictions):
            row = []
            for true_val, pred_val in zip(true_params, predicted_params):
                row.extend([f"{true_val:.4f}", f"{pred_val:.4f}"])
            writer.writerow(row)
    print(f"予測結果を {csv_path} に保存しました。")

def explain_prediction_grad_cam(model_path, input_data, target_param_index, layer_name):
    """Grad-CAM を用いて予測に対する入力データの影響を可視化する関数 (1D-CNN 用)"""
    model = tf.keras.models.load_model(model_path)

    # Grad-CAM の計算に必要なモデルを定義
    grad_model = tf.keras.models.Model(
        [[model.inputs]], [model.get_layer(layer_name).output, model.output] # 修正箇所
    )
    
    # 勾配を計算
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        loss = predictions[:, target_param_index]

    output = conv_outputs[0]  # 最初のサンプル
    grads = tape.gradient(loss, conv_outputs)[0] # 最初のサンプルの勾配

    # チャンネルごとの勾配の平均を計算
    pooled_grads = tf.reduce_mean(grads, axis=0)

    # 特徴マップに勾配の重みを適用
    heatmap = output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU を適用して、正の影響を持つ部分のみを強調
    heatmap = np.maximum(heatmap, 0)

    # ヒートマップを元の入力データのサイズにリサイズ
    interpolation = 'linear' if len(heatmap) < len(input_data[0]) else 'nearest'
    heatmap_resized = np.interp(np.linspace(0, 1, len(input_data[0])), np.linspace(0, 1, len(heatmap)), heatmap)

    # ヒートマップを 0 から 1 の範囲に正規化
    heatmap_resized /= np.max(heatmap_resized)

    print(heatmap_resized.shape)
    return heatmap_resized

def main(mode):
    # データのロード
    y_data, parameter_data = load_data(DATA_FILE)

    # データの準備
    # 波形データの形状を (サンプル数, 波形の長さ, 1) にリシェイプ
    y_data = np.expand_dims(y_data, axis=-1)
    num_params = parameter_data.shape[1]
    input_shape = (y_data.shape[1], 1)
    parameter_names = ["mu", "sigma", "offset", "amplitude", "exp_decay_rate", "exp_amplitude_factor"]

    # 訓練データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(y_data, parameter_data, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 訓練データからバリデーションデータを作成

    if mode == 1:
        # 既存のモデルを削除
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
            print("既存のモデルを削除しました。")
        else:
            print("既存のモデルは存在しません。")

        # モデルの作成と訓練
        print("モデルを新規に作成します。")
        model = create_model(input_shape, num_params)
        model.summary()
        print("モデルの訓練を開始します...")
        history = train_model(model, x_train, y_train, x_val, y_val, MODEL_SAVE_PATH)
        print("モデルの訓練が完了しました。")

        # モデルの評価
        print("モデルの評価を行います...")
        evaluate_model(MODEL_SAVE_PATH, x_test, y_test, PREDICTION_CSV_PATH)
        print("モデルの評価が完了しました。")

    elif mode == 2:
        # 保存済みモデルの評価
        print("保存済みモデルの評価を行います...")
        evaluate_model(MODEL_SAVE_PATH, x_test, y_test, PREDICTION_CSV_PATH)
        print("モデルの評価が完了しました。")

        # Grad-CAM の実行と可視化
        os.makedirs(GRAD_CAM_OUTPUT_DIR, exist_ok=True)
        print("\nGrad-CAMによる可視化を実行します...")
        num_visualize = 5  # 可視化するサンプル数
        layer_name = 'conv1d_2' # 可視化に使用する畳み込み層の名前 (修正)

        for i in np.random.choice(len(x_test), num_visualize, replace=False):
            sample_waveform = np.expand_dims(x_test[i], axis=0)
            true_params = y_test[i]

            for target_index, param_name in enumerate(parameter_names):
                heatmap = explain_prediction_grad_cam(MODEL_SAVE_PATH, sample_waveform, target_index, layer_name)

                # Grad-CAM の結果をプロット
                plt.figure(figsize=(10, 5))
                plt.plot(x_test[i].flatten(), label="Waveform")
                plt.plot(x_test[i].flatten(), heatmap, label="Grad-CAM", alpha=0.7)
                plt.xlabel("Data Point Index")
                plt.ylabel("Amplitude / Grad-CAM Score")
                plt.title(f"Grad-CAM Visualization for Parameter: {param_name}\nTrue Parameters: {true_params}")
                plt.legend()
                plt.tight_layout()
                output_path = os.path.join(GRAD_CAM_OUTPUT_DIR, f"grad_cam_{i}_{param_name}.png")
                plt.savefig(output_path)
                plt.close()
            print(f"サンプル {i} の Grad-CAM 可視化を {GRAD_CAM_OUTPUT_DIR} に保存しました。")


    print("\nTensorFlow が認識するデバイス:")
    print(tf.config.list_physical_devices())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model for waveform parameter prediction.")
    parser.add_argument('mode', type=int, choices=[1, 2], help='1: Train and evaluate, 2: Evaluate only')
    args = parser.parse_args()

    main(args.mode)
