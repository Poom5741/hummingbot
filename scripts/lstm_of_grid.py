from __future__ import annotations
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


parameters = {
    "model_url": "http://localhost:8000/predict",
    "model_seq_len": 20,
    "order_amount": Decimal("0.001"),
    "base_step": 0.002,
    "max_levels": 3,
    "imbalance_gate": 0.6,
    "cvd_gate": 0.0,
    "reseed_threshold": 0.01,
    "kill_switch": -0.05,
    "quote_cap": 3,
    "post_only": True,
}


class LSTMOFGrid(ScriptStrategyBase):
    """Grid strategy driven by an LSTM model."""

    markets = {"binance_paper_trade": {"BTC-USDT"}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        p = getattr(self, "parameters", parameters)
        self.model_url: str = p.get("model_url")
        self.model_seq_len: int = int(p.get("model_seq_len", 20))
        self.order_amount: Decimal = Decimal(str(p.get("order_amount", "0.001")))
        self.base_step: float = float(p.get("base_step", 0.002))
        self.max_levels: int = int(p.get("max_levels", 3))
        self.imbalance_gate: float = float(p.get("imbalance_gate", 0.6))
        self.cvd_gate: float = float(p.get("cvd_gate", 0.0))
        self.reseed_threshold: float = float(p.get("reseed_threshold", 0.01))
        self.kill_switch: float = float(p.get("kill_switch", -0.05))
        self.quote_cap: int = int(p.get("quote_cap", 3))
        self.post_only: bool = bool(p.get("post_only", True))

        self.exchange: str = next(iter(self.markets))
        self.trading_pair: str = next(iter(self.markets[self.exchange]))
        self.candles = CandlesFactory.get_candle(
            CandlesConfig(connector=self.exchange, trading_pair=self.trading_pair, interval="1m", max_records=1000)
        )
        self.candles.start()

        self.feature_buffer: pd.DataFrame = pd.DataFrame(
            columns=["ret", "atr", "rsi", "imbalance", "cvd", "spread", "microprice"]
        )
        self.last_reseed_price: Optional[Decimal] = None
        self.initial_value: Decimal = self._portfolio_value()
        self.killed: bool = False

    async def on_stop(self):
        self.candles.stop()

    def _portfolio_value(self) -> Decimal:
        connector = self.connectors[self.exchange]
        base, quote = self.trading_pair.split("-")
        mid_price = connector.get_mid_price(self.trading_pair) or Decimal("0")
        return connector.get_balance(quote) + connector.get_balance(base) * mid_price

    def _build_features(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df["ret"] = df["close"].pct_change()
        df["tr"] = df["high"].combine(df["close"].shift(), max) - df["low"].combine(df["close"].shift(), min)
        df["atr"] = df["tr"].rolling(14).mean()
        delta = df["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + rs)
        df["cvd"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()

        connector = self.connectors[self.exchange]
        ob = connector.get_order_book(self.trading_pair)
        bid = ob.best_bid_price
        ask = ob.best_ask_price
        bid_vol = ob.get_volume_for_price(True, bid)
        ask_vol = ob.get_volume_for_price(False, ask)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if bid_vol + ask_vol > 0 else 0
        spread = ask - bid
        micro = (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol) if bid_vol + ask_vol > 0 else (ask + bid) / 2

        last = df.iloc[-1]
        return pd.Series(
            {
                "ret": last["ret"],
                "atr": last["atr"],
                "rsi": last["rsi"],
                "imbalance": imbalance,
                "cvd": last["cvd"],
                "spread": spread,
                "microprice": micro,
            }
        )

    def _update_buffer(self, feats: pd.Series):
        self.feature_buffer = pd.concat([self.feature_buffer, feats.to_frame().T], ignore_index=True)
        if len(self.feature_buffer) > self.model_seq_len:
            self.feature_buffer = self.feature_buffer.iloc[-self.model_seq_len :]

    def _predict(self) -> Tuple[float, float]:
        data = self.feature_buffer.values.tolist()
        try:
            resp = httpx.post(self.model_url, json={"features": data}, timeout=1.0)
            resp.raise_for_status()
            js = resp.json()
            return float(js.get("p_up", 0.5)), float(js.get("vol_h", self.base_step))
        except Exception:
            momentum = float(self.feature_buffer["ret"].iloc[-1])
            atr = float(self.feature_buffer["atr"].iloc[-1])
            return 0.5 + np.tanh(momentum) * 0.1, max(self.base_step, atr)

    def _grid_from_prediction(self, p_up: float, vol_h: float) -> Tuple[List[Decimal], List[Decimal]]:
        mid = Decimal(str(self.connectors[self.exchange].get_mid_price(self.trading_pair)))
        bias = p_up - 0.5
        step = Decimal(str(max(self.base_step, vol_h)))
        levels = min(self.max_levels, max(1, int(vol_h / self.base_step)))
        bid_step = step * (1 + max(-bias, 0))
        ask_step = step * (1 + max(bias, 0))
        bids = [mid * (Decimal("1") - bid_step * Decimal(i + 1)) for i in range(levels)]
        asks = [mid * (Decimal("1") + ask_step * Decimal(i + 1)) for i in range(levels)]
        return bids, asks

    def _apply_order_flow_gates(self, bids: List[Decimal], asks: List[Decimal], feats: pd.Series):
        if feats["imbalance"] > self.imbalance_gate or feats["cvd"] > self.cvd_gate:
            asks = []
        if feats["imbalance"] < -self.imbalance_gate or feats["cvd"] < -self.cvd_gate:
            bids = []
        return bids, asks

    def _kill_switch_triggered(self) -> bool:
        val = self._portfolio_value()
        return (val / self.initial_value - Decimal("1")) < Decimal(str(self.kill_switch))

    def on_tick(self):
        if self.killed or not self.candles.ready:
            return
        mid = self.connectors[self.exchange].get_mid_price(self.trading_pair)
        if mid is None:
            return

        feats = self._build_features(self.candles.candles_df)
        self._update_buffer(feats)
        if len(self.feature_buffer) < self.model_seq_len:
            return

        if self._kill_switch_triggered():
            self.cancel_all_orders()
            self.killed = True
            self.logger().warning("Kill switch triggered; stopping trades.")
            return

        if self.last_reseed_price is None:
            self.last_reseed_price = mid
        elif abs(mid - self.last_reseed_price) / self.last_reseed_price > Decimal(str(self.reseed_threshold)):
            self.cancel_all_orders()
            self.last_reseed_price = mid

        p_up, vol_h = self._predict()
        bids, asks = self._grid_from_prediction(p_up, vol_h)
        bids, asks = self._apply_order_flow_gates(bids, asks, feats)
        bids = bids[: self.quote_cap]
        asks = asks[: self.quote_cap]

        self.cancel_all_orders()
        self._place_orders(bids, True)
        self._place_orders(asks, False)

    def _place_orders(self, prices: List[Decimal], is_buy: bool):
        connector = self.connectors[self.exchange]
        rule = connector.trading_rules.get(self.trading_pair)
        otype = OrderType.LIMIT_MAKER if self.post_only else OrderType.LIMIT
        for idx, price in enumerate(prices):
            q_price = connector.quantize_order_price(self.trading_pair, price)
            q_amount = connector.quantize_order_amount(self.trading_pair, self.order_amount)
            if rule and q_amount * q_price < rule.min_notional_size:
                continue
            cid = f"lstm-grid-{'buy' if is_buy else 'sell'}-{idx}"
            if is_buy:
                self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=q_amount,
                    order_type=otype,
                    price=q_price,
                    client_order_id=cid,
                    post_only=self.post_only,
                )
            else:
                self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=q_amount,
                    order_type=otype,
                    price=q_price,
                    client_order_id=cid,
                    post_only=self.post_only,
                )

    def cancel_all_orders(self):
        for o in self.get_active_orders(self.exchange):
            self.cancel(self.exchange, o.trading_pair, o.client_order_id)

