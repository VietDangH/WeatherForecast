import sys
# Force UTF-8 stdout/stderr so the Vietnamese log messages don't crash on
# Windows consoles that default to cp1252 (UnicodeEncodeError).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import tensorflow as tf
import numpy as np

import src.config as config
import src.data_prep as data_prep
import src.lstm_model as lstm_model
import src.evaluate as evaluate
import src.diagnostics as diagnostics

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Lỗi cấu hình GPU: {e}")
    else:
        print("Không tìm thấy GPU. Hệ thống sẽ chạy bằng CPU.")

    df = data_prep.load_and_preprocess_data()
    train_X_s, train_y_s, val_X_s, val_y_s, test_X_s, test_y_s, scaler_y = data_prep.split_and_scale(df)

    X_train, y_train = data_prep.create_windows(train_X_s, train_y_s, config.LOOKBACK, config.HORIZON, step=1)
    X_val, y_val = data_prep.create_windows(val_X_s, val_y_s, config.LOOKBACK, config.HORIZON, step=1)
    X_test, y_test = data_prep.create_windows(test_X_s, test_y_s, config.LOOKBACK, config.HORIZON, step=24)

    print(f"\n🔥 BẮT ĐẦU TRAINING VỚI ADAPTIVE LR: {config.BEST_PARAMS}")
    
    all_seed_predictions = {}
    histories = {}
    N_win = y_test.shape[0]

    for seed in config.SEEDS:
        print(f"\n---> ĐANG HUẤN LUYỆN SEED: {seed} <---")
        model, history = lstm_model.build_and_train_model(X_train, y_train, X_val, y_val, seed)
        histories[seed] = history
        
        y_pred_real = lstm_model.predict_and_inverse(model, X_test, scaler_y, N_win)
        all_seed_predictions[seed] = y_pred_real

    ensemble_pred = np.mean([all_seed_predictions[s] for s in config.SEEDS], axis=0)
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, len(config.TARGETS))).reshape(N_win, config.HORIZON, len(config.TARGETS))

    evaluate.evaluate_metrics(y_test_real, ensemble_pred)
    diagnostics.run_diagnostics(y_test_real, ensemble_pred)
    diagnostics.plot_all_results(y_test_real, ensemble_pred, histories)

    print(f"\n [HOÀN TẤT] Toàn bộ file CSV và Biểu đồ ảnh đã được lưu vào thư mục: {config.OUTPUT_DIR}")