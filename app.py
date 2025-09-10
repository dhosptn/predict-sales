from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)

    required_cols = {"date", "product", "sales"}
    if not required_cols.issubset(df.columns):
        return jsonify({"error": f"Data harus memiliki kolom {required_cols}"}), 400

    df['date'] = pd.to_datetime(df['date'])
    results = []

    for product, group in df.groupby("product"):
        group = group.sort_values('date')
        sales_series = group['sales'].astype(float).values
        last_date = group['date'].max()

        if len(sales_series) < 3:
            # Data terlalu sedikit → pakai rata-rata
            forecast = [int(round(np.mean(sales_series)))] * 7
        else:
            try:
                if len(sales_series) >= 14:  
                    # Kalau data minimal 2 minggu → tambahkan musiman mingguan
                    model = ExponentialSmoothing(
                        sales_series,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=7,
                        initialization_method="estimated"
                    )
                else:
                    # Kalau data pendek → hanya trend
                    model = ExponentialSmoothing(
                        sales_series,
                        trend="add",
                        seasonal=None,
                        initialization_method="estimated"
                    )

                fit_model = model.fit(optimized=True)
                forecast = fit_model.forecast(7)
                forecast = [int(round(x)) for x in forecast]
            except Exception:
                forecast = [int(round(np.mean(sales_series)))] * 7

        # Buat detail tanggal prediksi
        future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]
        forecast_detail = [{"date": future_dates[i], "predicted_sales": forecast[i]} for i in range(7)]

        # Deteksi tren
        if forecast[-1] > forecast[0]:
            trend = "Meningkat"
        elif forecast[-1] < forecast[0]:
            trend = "Menurun"
        else:
            trend = "Stabil"

        # Rekomendasi stok → 15% di atas rata-rata prediksi
        recommended_stock = int(np.mean(forecast) * 1.15)

        results.append({
            "product": product,
            "forecast": forecast,
            "forecast_detail": forecast_detail,
            "trend": trend,
            "recommended_stock": recommended_stock
        })

    return jsonify({"products": results})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
