from exact_hidden_markov_models import RF_HMM, RF_HMM_LSTM, Encoder_RF_HMM
import numpy as np

def generate_time_series(length: int, vector_dim: int) -> np.ndarray:
    return np.random.multivariate_normal(
        mean=np.zeros(vector_dim),
        cov=np.eye(vector_dim, vector_dim),
        size=length
    )

def test_rf_hmm() -> None:
    model = RF_HMM(n_states=2)
    model.train(generate_time_series(1000, 2))
    params = model.eval(generate_time_series(100, 2))

    print('test rf_hmm successful')

def test_rf_hmm_lstm() -> None:
    model = RF_HMM_LSTM(n_states=2, T=10)
    model.train(
        X=generate_time_series(1000, 4),
        T=10,
        Y=generate_time_series(1000, 4),
        epochs=5
    )
    params = model.predict(
        X=generate_time_series(100, 4),
        T=10
    )

    print('test rf_hmm_lstm successful')

def test_encoder_rf_hmm() -> None:
    model = Encoder_RF_HMM(n_states=2, input_dim=(10, 6))
    model.train(
        X=generate_time_series(1000, 6),
        T=10,
        epochs=5
    )
    params = model.predict(
        X=generate_time_series(100, 6),
        T=10
    )

    print('test encoder_rf_hmm successful')