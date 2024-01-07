import matplotlib.pyplot as plt
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS
from neuralforecast.utils import AirPassengersDF


def main() -> None:
    y_df = AirPassengersDF
    horizon = 60
    models = [
        LSTM(
            h=horizon,
            max_steps=500,
            scaler_type="standard",
            encoder_hidden_size=64,
            decoder_hidden_size=64,
        ),
        NHITS(
            h=horizon,
            input_size=2 * horizon,
            max_steps=100,
            n_freq_downsample=[2, 1, 1],
        ),
    ]
    nf = NeuralForecast(models=models, freq="M")
    nf.fit(df=y_df)
    y_hat_df = nf.predict()
    y_hat_df = y_hat_df.reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    plot_df = pd.concat([y_df, y_hat_df]).set_index("ds")
    plot_df[["y", "LSTM", "NHITS"]].plot(ax=ax, linewidth=2)
    ax.set_title("Forecast Air Passenger Number", fontsize=22)
    ax.set_ylabel("Monthly Passenger Number", fontsize=20)
    ax.set_xlabel("Time", fontsize=20)
    ax.legend(prop={"size": 15})
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
