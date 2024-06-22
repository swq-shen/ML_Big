from feat import FeatureGenerator
from model import Model
from dl import DataLoader, sym
from tradingcalendar import Calendar
import torch


def ttt(stock):
    epochs = 1
    calendar = Calendar()
    ft = FeatureGenerator(None)
    dloader = DataLoader("../archive", stock)
    dates = calendar.range(20230601, 20231130)
    rawData = dloader.loadDates(dates)
    print(rawData.columns)
    X, y = ft.genFeatures(rawData)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo = Model(device, stock)
    try:
        mo.load_net('model.pth')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model. Training new model instead. Error: {e}")
    mo.fit(X, y, epochs)
    mo.save_net('model.pth')
    print(stock)


if __name__ == '__main__':
    symbol = sym()
    for stock in symbol:
        ttt(stock)
