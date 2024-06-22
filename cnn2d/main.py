import torch
import matplotlib.pyplot as plt
from dl import sym
from model import Model


def pre(stock, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo = Model(device, stock)
    t_dates = mo.calendar.range(20231201, 20231229)
    t_rawData = mo.dloader.loadDates(t_dates)
    t_xdf, t_ydf = mo.featGenerator.genFeatures(t_rawData)

    num_features = t_xdf.shape[1]
    num_t = t_xdf.shape[0]
    t_xdf = mo.time_series_to_images(t_xdf, num_features=num_features)
    t_ydf = t_ydf[:num_t - mo.time + 1]
    t_xdf = torch.tensor(t_xdf, dtype=torch.float32).to(device)
    try:
        mo.load_net(path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model. Training new model instead. Error: {e}")
    p, r, e, _ = mo.tes(t_xdf, t_ydf)
    return p, r, e


def show_(pp, rr, ee):
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pp, rr, ee)
    ax.set_title('3D')
    ax.set_xlabel('Pearson ')
    ax.set_ylabel('R2')
    ax.set_zlabel('MSE')
    plt.show()


if __name__ == '__main__':
    pp = []
    rr = []
    ee = []
    t_num = 100
    path = 'good/model_1.pth'
    symbol = sym()
    for stock in symbol:
        p, r, e = pre(stock, path)
        pp.append(p)
        rr.append(r)
        ee.append(e)
    show_(pp, rr, ee)
    show_(pp[:t_num], rr[:t_num], ee[:t_num])
    print(pp)
    print(rr)
    print(ee)
