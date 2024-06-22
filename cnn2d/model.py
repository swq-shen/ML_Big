import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dl import DataLoader
from eval import MeowEvaluator
from feat import FeatureGenerator
from tradingcalendar import Calendar
from eval import composite_loss


class ResNLS(nn.Module):
    def __init__(self, input_channels, height=12, hidden_size=16, output_size=1):
        super(ResNLS, self).__init__()
        # 使用二维卷积层
        self.cnn1 = nn.Conv2d(in_channels=input_channels,
                              out_channels=hidden_size,
                              kernel_size=(4, height), stride=(1, 1), padding=(1, 0))
        self.cnn2 = nn.Conv2d(in_channels=hidden_size,
                              out_channels=hidden_size,
                              kernel_size=(4, 1), stride=(1, 1), padding=(1, 0))
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=hidden_size * 3,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.li = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.cnn1(x)
        x = F.relu(x)
        # x = self.cnn2(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1)
        seq_len = height * width
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        # x = self.dropout(x)
        x = self.li(x)
        x = self.linear(x)
        return x


class Model(object):
    def __init__(self, device, stock, model=None):
        self.device = device
        self.time = 11
        if model is None:
            self.model = ResNLS(input_channels=1, hidden_size=64, output_size=1).to(device)
        else:
            self.model = model.to(device)
        self.evaluator = MeowEvaluator("../archive")
        self.calendar = Calendar()
        self.dloader = DataLoader("../archive", stock)
        self.featGenerator = FeatureGenerator("../archive")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

    def fit(self, xdf, ydf, epochs=1, batch_size=64):
        t_dates = self.calendar.range(20231201, 20231229)
        t_rawData = self.dloader.loadDates(t_dates)
        t_xdf, t_ydf = self.featGenerator.genFeatures(t_rawData)

        num_features = xdf.shape[1]
        num_x = xdf.shape[0]
        num_t = t_xdf.shape[0]
        images = self.time_series_to_images(xdf, num_features=num_features)
        images_tensor = torch.tensor(images, dtype=torch.float32).to(self.device)
        ydf = ydf[:num_x - self.time+1]
        ydf = torch.tensor(ydf.values, dtype=torch.float32).to(self.device)

        t_xdf = self.time_series_to_images(t_xdf, num_features=num_features)
        t_ydf = t_ydf[:num_t - self.time+1]
        t_xdf = torch.tensor(t_xdf, dtype=torch.float32).to(self.device)
        criterion = composite_loss
        self.model.train()
        num_batches = images_tensor.size(0) // batch_size
        for epoch in range(epochs):
            print(epoch)
            epoch_loss = 0
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                batch_x = images_tensor[start_idx:end_idx]
                batch_y = ydf[start_idx:end_idx]
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                p, r, e, y = self.tes(t_xdf, t_ydf)
                epoch_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / num_batches}")
        print("Training complete")

    def predict(self, xdf, batch_size=64):
        self.model.eval()
        with torch.no_grad():
            predictions_list = []
            num_batches = (xdf.size(0) + batch_size - 1) // batch_size
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, len(xdf))
                batch_x = xdf[start_idx:end_idx].to(self.device)
                batch_predictions = self.model(batch_x)
                predictions_list.append(batch_predictions.cpu().numpy())
            predictions = np.concatenate(predictions_list, axis=0)
        return predictions

    def tes(self, xdf, ydf):
        predictions = self.predict(xdf)
        predictions = predictions.flatten()
        predictions_series = pd.Series(predictions, index=ydf.index)
        if "forecast" not in ydf.columns:
            ydf["forecast"] = predictions_series
        else:
            ydf.loc[:, "forecast"] = predictions_series
        p, r, e = self.evaluator.eval(ydf)
        return p, r, e, ydf

    def save_net(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_net(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def time_series_to_images(self, xdf, num_features):
        num_images = (xdf.shape[0] - self.time + 1)
        images = []
        for i in range(num_images):
            image_data = xdf.iloc[i:i + self.time].values
            image_reshaped = image_data.reshape((1, self.time, num_features))
            images.append(image_reshaped)
        images = np.array(images)
        return images



