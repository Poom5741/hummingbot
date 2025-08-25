import torch
from fastapi import FastAPI
from pydantic import BaseModel


class LSTMModel(torch.nn.Module):
    """Minimal LSTM model with two outputs."""

    def __init__(self, input_size: int = 1, hidden_size: int = 10):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.linear(output)


model = LSTMModel()
model.eval()


class PredictRequest(BaseModel):
    data: list[float]


app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest):
    """Return p_up and vol_h predictions for provided sequence."""
    x = torch.tensor(request.data, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        pred = model(x).numpy()[0]
    return {"p_up": float(pred[0]), "vol_h": float(pred[1])}
