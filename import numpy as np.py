import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

def generate_telemetry(n_steps=600, anomaly_ratio=0.06, freq_minutes=15):
    time = pd.date_range(start="2025-09-01", periods=n_steps, freq=f"{freq_minutes}T")
    battery = 3.7 + 0.02*np.sin(np.linspace(0, 6*np.pi, n_steps)) + 0.005*np.random.randn(n_steps)
    temperature = 20 + 2*np.sin(np.linspace(0, 3*np.pi, n_steps)) + 0.3*np.random.randn(n_steps)
    signal = 100 + 5*np.cos(np.linspace(0, 4*np.pi, n_steps)) + np.random.randn(n_steps)
    solar = 0.9 + 0.05*np.sin(np.linspace(0, 8*np.pi, n_steps)) + 0.01*np.random.randn(n_steps)

    df = pd.DataFrame({
        "timestamp": time,
        "battery_v": battery,
        "temperature_c": temperature,
        "signal_db": signal,
        "solar_eff": solar
    }).set_index("timestamp")

    n_anom = int(n_steps * anomaly_ratio)
    anomaly_indices = np.random.choice(n_steps, n_anom, replace=False)
    for idx in anomaly_indices:
        typ = np.random.choice(["drop_power", "overheat", "signal_loss", "solar_drop"])
        if typ == "drop_power":
            df.iloc[idx:idx+3, df.columns.get_loc("battery_v")] -= 0.5 + 0.1*np.random.rand()
        elif typ == "overheat":
            df.iloc[idx:idx+2, df.columns.get_loc("temperature_c")] += 6 + 2*np.random.rand()
        elif typ == "signal_loss":
            df.iloc[idx:idx+4, df.columns.get_loc("signal_db")] -= 15 + 5*np.random.rand()
        elif typ == "solar_drop":
            df.iloc[idx:idx+3, df.columns.get_loc("solar_eff")] -= 0.4 + 0.1*np.random.rand()
    return df, anomaly_indices

def create_sequences(data, seq_len=20):
    X = []
    for i in range(len(data) - seq_len + 1):
        X.append(data[i:i+seq_len])
    return np.array(X)

def main():
    SEQ_LEN = 20
    df, true_anom_idx = generate_telemetry(n_steps=600, anomaly_ratio=0.06)
    print(f"Samples: {len(df)}; injected anomalies: {len(true_anom_idx)}")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    X = create_sequences(data_scaled, seq_len=SEQ_LEN)
    print("Sequence shape:", X.shape)

    use_tf = True
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
        tf.get_logger().setLevel('ERROR')
    except:
        use_tf = False

    if use_tf:
        n_features = X.shape[2]
        inputs = Input(shape=(SEQ_LEN, n_features))
        x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
        x = LSTM(32, activation='tanh', return_sequences=False)(x)
        bottleneck = RepeatVector(SEQ_LEN)(Dense(8, activation='tanh')(x))
        x = LSTM(32, activation='tanh', return_sequences=True)(bottleneck)
        x = LSTM(64, activation='tanh', return_sequences=True)(x)
        outputs = TimeDistributed(Dense(n_features))(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')

        n_train = int(0.7 * len(X))
        model.fit(X[:n_train], X[:n_train], epochs=8, batch_size=32, validation_split=0.1, verbose=1)

        X_pred = model.predict(X, verbose=0)
        mse = np.mean(np.mean(np.square(X_pred - X), axis=2), axis=1)
        anomaly_scores = mse
        train_mse = mse[:n_train]
        threshold = train_mse.mean() + 3*train_mse.std()
    else:
        df_z = (df - df.mean()) / df.std()
        combined_z = np.sqrt((df_z**2).sum(axis=1))
        rolling_score = combined_z.rolling(window=SEQ_LEN, min_periods=1).mean()
        anomaly_scores = (rolling_score - rolling_score.min()) / (rolling_score.max() - rolling_score.min())
        anomaly_scores = anomaly_scores.values[SEQ_LEN-1:]
        threshold = 0.6

    seq_timestamps = df.index[SEQ_LEN-1:]
    scores_series = pd.Series(anomaly_scores, index=seq_timestamps)
    flags = scores_series > threshold
    print(f"Detected anomalous sequences: {flags.sum()} / {len(flags)} (threshold={threshold:.6f})")

    detected = scores_series[flags]
    print("\nSample detected windows:")
    print(detected.head(10))

    plt.figure(figsize=(12,4))
    plt.plot(scores_series.index, scores_series.values, label="anomaly_score")
    plt.axhline(threshold, linestyle='--', label='threshold')
    plt.title("Anomaly Score over Time (sequence-wise)")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,3))
    plt.plot(df.index, df["battery_v"].values, label="battery_v")
    for t in scores_series[flags].index:
        start = t - pd.Timedelta(minutes=15*SEQ_LEN)
        plt.axvspan(start, t, alpha=0.12)
    plt.title("Battery Voltage with Detected Anomaly Windows (shaded)")
    plt.xlabel("Time")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    detected_df = pd.DataFrame({"timestamp": detected.index, "anomaly_score": detected.values}).set_index("timestamp")
    detected_df.to_csv("detected_anomalies.csv")
    print("Saved detected windows to detected_anomalies.csv")

if __name__ == "__main__":
    main()
