# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Provides data loaders for historical Polymarket data from various APIs.
"""

from __future__ import annotations

from typing import Any

import msgspec
import pandas as pd
import requests

from nautilus_trader.adapters.polymarket.common.constants import POLYMARKET_HTTP_RATE_LIMIT
from nautilus_trader.adapters.polymarket.common.parsing import parse_polymarket_instrument
from nautilus_trader.adapters.polymarket.common.symbol import get_polymarket_token_id
from nautilus_trader.core import nautilus_pyo3
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.datetime import millis_to_nanos, nanos_to_millis
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.data import OrderBookDelta
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BookAction
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.instruments import BinaryOption


class PolymarketDataLoader:
    """
    Provides a data loader for historical Polymarket market data.

    This loader fetches data from:
    - Polymarket Gamma API (market and event information)
    - Polymarket CLOB API (market details, price/trade history, and orderbook history)

    If no `http_client` is provided, the loader creates one with a default rate limit
    of 100 requests per minute, matching Polymarket's public endpoint limit.

    Parameters
    ----------
    instrument : BinaryOption
        The binary option instrument to load data for.
    token_id : str, optional
        The Polymarket token ID for this instrument.
    http_client : nautilus_pyo3.HttpClient, optional
        The HTTP client to use for requests. If not provided, a new client will be created.

    """

    def __init__(
        self,
        instrument: BinaryOption,
        token_id: str | None = None,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> None:
        self._instrument = instrument
        self._token_id = token_id
        self._http_client = http_client or self._create_http_client()

    @staticmethod
    def _create_http_client() -> nautilus_pyo3.HttpClient:
        return nautilus_pyo3.HttpClient(
            default_quota=nautilus_pyo3.Quota.rate_per_minute(POLYMARKET_HTTP_RATE_LIMIT),
        )

    @staticmethod
    async def _fetch_market_by_slug(
        slug: str,
        http_client: nautilus_pyo3.HttpClient,
    ) -> dict[str, Any]:
        PyCondition.valid_string(slug, "slug")

        response = await http_client.get(
            url=f"https://gamma-api.polymarket.com/markets/slug/{slug}",
        )

        if response.status == 404:
            raise ValueError(f"Market with slug '{slug}' not found")

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        data = msgspec.json.decode(response.body)

        if isinstance(data, list):
            if not data:
                raise ValueError(f"Market with slug '{slug}' not found")
            market = data[0]
        else:
            market = data

        if not isinstance(market, dict):
            raise RuntimeError(
                f"Unexpected response type for slug '{slug}': {type(market).__name__}",
            )

        return market

    @staticmethod
    async def _fetch_market_details(
        condition_id: str,
        http_client: nautilus_pyo3.HttpClient,
    ) -> dict[str, Any]:
        PyCondition.valid_string(condition_id, "condition_id")

        response = await http_client.get(
            url=f"https://clob.polymarket.com/markets/{condition_id}",
        )

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        return msgspec.json.decode(response.body)

    @staticmethod
    async def _fetch_event_by_slug(
        slug: str,
        http_client: nautilus_pyo3.HttpClient,
    ) -> dict[str, Any]:
        PyCondition.valid_string(slug, "slug")

        response = await http_client.get(
            url="https://gamma-api.polymarket.com/events",
            params={"slug": slug},
        )

        if response.status == 404:
            raise ValueError(f"Event with slug '{slug}' not found")

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        events = msgspec.json.decode(response.body)

        if not events:
            raise ValueError(f"Event with slug '{slug}' not found")

        return events[0]

    @classmethod
    async def from_market_slug(
        cls,
        slug: str,
        token_index: int = 0,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> PolymarketDataLoader:
        """
        Create a loader by fetching market data from Polymarket APIs.

        Parameters
        ----------
        slug : str
            The market slug to search for.
        token_index : int, default 0
            The index of the token to use (0 for first outcome, 1 for second).
        http_client : nautilus_pyo3.HttpClient, optional
            The HTTP client to use for requests. If not provided, a new client will be created.

        Returns
        -------
        PolymarketDataLoader

        Raises
        ------
        ValueError
            If market with slug is not found or has no tokens.
        RuntimeError
            If HTTP requests fail.

        """
        client = http_client or cls._create_http_client()
        market = await cls._fetch_market_by_slug(slug, client)
        condition_id = market["conditionId"]
        market_details = await cls._fetch_market_details(condition_id, client)
        tokens = market_details.get("tokens", [])

        if not tokens:
            raise ValueError(f"No tokens found for market: {condition_id}")

        if token_index >= len(tokens):
            raise ValueError(
                f"Token index {token_index} out of range (market has {len(tokens)} tokens)",
            )

        token = tokens[token_index]
        token_id = token["token_id"]
        outcome = token["outcome"]

        instrument = parse_polymarket_instrument(
            market_info=market_details,
            token_id=token_id,
            outcome=outcome,
        )

        return cls(instrument=instrument, token_id=token_id, http_client=client)

    @classmethod
    async def from_event_slug(
        cls,
        slug: str,
        token_index: int = 0,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> list[PolymarketDataLoader]:
        """
        Create loaders for all markets in an event.

        This is useful for events that contain multiple related markets,
        such as temperature bucket markets where each bucket is a separate market.

        Parameters
        ----------
        slug : str
            The event slug to fetch.
        token_index : int, default 0
            The index of the token to use (0 for first outcome, 1 for second).
        http_client : nautilus_pyo3.HttpClient, optional
            The HTTP client to use for requests. If not provided, a new client will be created.

        Returns
        -------
        list[PolymarketDataLoader]
            List of loaders, one for each market in the event.

        Raises
        ------
        ValueError
            If event with slug is not found, has no markets, or token_index is out of range.

        """
        client = http_client or cls._create_http_client()
        event = await cls._fetch_event_by_slug(slug, client)
        markets = event.get("markets", [])

        if not markets:
            raise ValueError(f"No markets found in event '{slug}'")

        loaders: list[PolymarketDataLoader] = []

        for market in markets:
            condition_id = market.get("conditionId")
            if not condition_id:
                continue

            market_details = await cls._fetch_market_details(condition_id, client)

            tokens = market_details.get("tokens", [])
            if not tokens:
                continue

            if token_index >= len(tokens):
                raise ValueError(
                    f"Token index {token_index} out of range "
                    f"(market {condition_id} has {len(tokens)} tokens)",
                )

            token = tokens[token_index]
            token_id = token["token_id"]
            outcome = token["outcome"]

            instrument = parse_polymarket_instrument(
                market_info=market_details,
                token_id=token_id,
                outcome=outcome,
            )

            loaders.append(cls(instrument=instrument, token_id=token_id, http_client=client))

        return loaders

    @staticmethod
    async def query_market_by_slug(
        slug: str,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> dict[str, Any]:
        """
        Query market data by slug without requiring a loader instance.

        Parameters
        ----------
        slug : str
            The market slug to fetch.
        http_client : nautilus_pyo3.HttpClient, optional
            The HTTP client to use for the request.

        Returns
        -------
        dict[str, Any]
            Market data dictionary.

        Raises
        ------
        ValueError
            If market with the given slug is not found.
        RuntimeError
            If HTTP request fails.

        """
        client = http_client or PolymarketDataLoader._create_http_client()
        return await PolymarketDataLoader._fetch_market_by_slug(slug, client)

    @staticmethod
    async def query_market_details(
        condition_id: str,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> dict[str, Any]:
        """
        Query detailed market information without requiring a loader instance.

        Parameters
        ----------
        condition_id : str
            The market condition ID.
        http_client : nautilus_pyo3.HttpClient, optional
            The HTTP client to use for the request.

        Returns
        -------
        dict[str, Any]
            Detailed market information.

        Raises
        ------
        RuntimeError
            If HTTP request fails.

        """
        client = http_client or PolymarketDataLoader._create_http_client()
        return await PolymarketDataLoader._fetch_market_details(condition_id, client)

    @staticmethod
    async def query_event_by_slug(
        slug: str,
        http_client: nautilus_pyo3.HttpClient | None = None,
    ) -> dict[str, Any]:
        """
        Query event data by slug without requiring a loader instance.

        Parameters
        ----------
        slug : str
            The event slug to fetch.
        http_client : nautilus_pyo3.HttpClient, optional
            The HTTP client to use for the request.

        Returns
        -------
        dict[str, Any]
            Event data dictionary containing 'markets' array and event metadata.

        Raises
        ------
        ValueError
            If event with the given slug is not found.
        RuntimeError
            If HTTP request fails.

        """
        client = http_client or PolymarketDataLoader._create_http_client()
        return await PolymarketDataLoader._fetch_event_by_slug(slug, client)

    @property
    def instrument(self) -> BinaryOption:
        """
        Return the instrument for this loader.
        """
        return self._instrument

    @property
    def token_id(self) -> str | None:
        """
        Return the token ID for this loader.
        """
        return self._token_id

    async def load_orderbook_snapshots(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        limit: int = 500,
    ) -> list[OrderBookDeltas]:
        """
        Load orderbook snapshots for the loader's instrument.

        This is a convenience method that fetches and parses orderbook history
        using the loader's stored token_id.

        Parameters
        ----------
        start : pd.Timestamp
            Start time for query window.
        end : pd.Timestamp
            End time for query window.
        limit : int, default 500
            Number of snapshots per request (max 500).

        Returns
        -------
        list[OrderBookDeltas]
            Parsed orderbook deltas ready for backtesting.

        Raises
        ------
        ValueError
            If token_id was not provided during initialization.

        """
        if self._token_id is None:
            raise ValueError(
                "token_id is required for this method. "
                "Use from_market_slug() to create a loader with token_id, "
                "or pass token_id to __init__()",
            )

        # Convert timestamps to milliseconds for the API
        start_time_ms = int(start.timestamp() * 1000)
        end_time_ms = int(end.timestamp() * 1000)

        snapshots = await self.fetch_orderbook_history(
            token_id=self._token_id,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=limit,
        )

        return self.parse_orderbook_snapshots(snapshots)

    async def load_trades(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fidelity: int = 1,
    ) -> list[TradeTick]:
        """
        Load synthetic trade ticks from price history for the loader's instrument.

        This is a convenience method that fetches and parses price history
        using the loader's stored token_id.

        Parameters
        ----------
        start : pd.Timestamp
            Start time for range.
        end : pd.Timestamp
            End time for range.
        fidelity : int, default 1
            Data resolution in minutes.

        Returns
        -------
        list[TradeTick]
            Parsed trade ticks ready for backtesting.

        Raises
        ------
        ValueError
            If token_id was not provided during initialization.

        """
        if self._token_id is None:
            raise ValueError(
                "token_id is required for this method. "
                "Use from_market_slug() to create a loader with token_id, "
                "or pass token_id to __init__()",
            )

        # Convert timestamps to milliseconds for the API
        start_time_ms = int(start.timestamp() * 1000)
        end_time_ms = int(end.timestamp() * 1000)

        history = await self.fetch_price_history(
            token_id=self._token_id,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            fidelity=fidelity,
        )

        return self.parse_price_history(history)

    async def fetch_event_by_slug(self, slug: str) -> dict[str, Any]:
        """
        Fetch an event by slug from the Polymarket Gamma API.

        Events contain multiple markets (e.g., temperature bucket markets
        are grouped under a single event like "highest-temperature-in-nyc-on-january-26").

        Parameters
        ----------
        slug : str
            The event slug to fetch.

        Returns
        -------
        dict[str, Any]
            Event data dictionary containing 'markets' array and event metadata.

        Raises
        ------
        ValueError
            If event with the given slug is not found.
        RuntimeError
            If HTTP requests fail.

        """
        return await self._fetch_event_by_slug(slug, self._http_client)

    async def fetch_events(
        self,
        active: bool = True,
        closed: bool = False,
        archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Fetch events from Polymarket Gamma API.

        Parameters
        ----------
        active : bool, default True
            Filter for active events.
        closed : bool, default False
            Include closed events.
        archived : bool, default False
            Include archived events.
        limit : int, default 100
            Maximum number of events to return.
        offset : int, default 0
            Offset for pagination.

        Returns
        -------
        list[dict[str, Any]]
            List of event data dictionaries.

        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "archived": str(archived).lower(),
            "limit": str(limit),
            "offset": str(offset),
        }
        response = await self._http_client.get(
            url="https://gamma-api.polymarket.com/events",
            params=params,
        )

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        return msgspec.json.decode(response.body)

    async def get_event_markets(self, slug: str) -> list[dict[str, Any]]:
        """
        Get all markets within an event by slug.

        This is a convenience method that fetches an event and extracts its markets.

        Parameters
        ----------
        slug : str
            The event slug to fetch markets from.

        Returns
        -------
        list[dict[str, Any]]
            List of market dictionaries within the event.

        Raises
        ------
        ValueError
            If event with the given slug is not found.

        """
        event = await self.fetch_event_by_slug(slug)
        return event.get("markets", [])

    async def fetch_markets(
        self,
        active: bool = True,
        closed: bool = False,
        archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Fetch markets from Polymarket Gamma API.

        Parameters
        ----------
        active : bool, default True
            Filter for active markets.
        closed : bool, default False
            Include closed markets.
        archived : bool, default False
            Include archived markets.
        limit : int, default 100
            Maximum number of markets to return.
        offset : int, default 0
            Offset for pagination.

        Returns
        -------
        list[dict]
            List of market data dictionaries.

        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "archived": str(archived).lower(),
            "limit": str(limit),
            "offset": str(offset),
        }
        response = await self._http_client.get(
            url="https://gamma-api.polymarket.com/markets",
            params=params,
        )

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        return msgspec.json.decode(response.body)

    async def fetch_market_by_slug(self, slug: str) -> dict[str, Any]:
        """
        Fetch a single market by slug using the Polymarket Gamma API slug endpoint.

        Parameters
        ----------
        slug : str
            The market slug to fetch.

        Returns
        -------
        dict[str, Any]
            Market data dictionary.

        Raises
        ------
        ValueError
            If market with the given slug is not found.
        RuntimeError
            If HTTP requests fail.

        """
        return await self._fetch_market_by_slug(slug, self._http_client)

    async def find_market_by_slug(self, slug: str) -> dict[str, Any]:
        """
        Find a specific market by slug.

        Parameters
        ----------
        slug : str
            The market slug to search for.

        Returns
        -------
        dict[str, Any]
            Market data dictionary.

        Raises
        ------
        ValueError
            If market with the given slug is not found.

        """
        return await self.fetch_market_by_slug(slug)

    async def fetch_market_details(self, condition_id: str) -> dict[str, Any]:
        """
        Fetch detailed market information from Polymarket CLOB API.

        Parameters
        ----------
        condition_id : str
            The market condition ID.

        Returns
        -------
        dict[str, Any]
            Detailed market information.

        """
        return await self._fetch_market_details(condition_id, self._http_client)

    async def fetch_orderbook_history(
        self,
        token_id: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Fetch orderbook history from Polymarket CLOB API.

        Parameters
        ----------
        token_id : str
            The Polymarket asset/token identifier.
        start_time_ms : int
            Unix timestamp in milliseconds for query window start.
        end_time_ms : int
            Unix timestamp in milliseconds for query window end.
        limit : int, default 500
            Number of snapshots per request (max 500).

        Returns
        -------
        list[dict[str, Any]]
            List of orderbook snapshot dictionaries.

        Notes
        -----
        This method automatically handles pagination using offset-based requests.

        """
        PyCondition.valid_string(token_id, "token_id")

        all_snapshots = []
        offset = 0

        while True:
            params = {
                "asset_id": token_id,
                "startTs": start_time_ms,
                "endTs": end_time_ms,
                "limit": limit,
                "offset": offset,
            }

            response = await self._http_client.get(
                url="https://clob.polymarket.com/orderbook-history",
                params=params,
            )

            if response.status != 200:
                raise RuntimeError(
                    f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
                )

            data = msgspec.json.decode(response.body)

            snapshots = data.get("data", [])
            all_snapshots.extend(snapshots)

            total_count = data.get("count", 0)
            offset += len(snapshots)

            if offset >= total_count or len(snapshots) < limit:
                break

        return all_snapshots

    async def fetch_price_history(
        self,
        token_id: str,
        start_time_ms: int,
        end_time_ms: int,
        fidelity: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Fetch price history from Polymarket CLOB API.

        Parameters
        ----------
        token_id : str
            The market/token identifier.
        start_time_ms : int
            Unix timestamp in milliseconds for range start.
        end_time_ms : int
            Unix timestamp in milliseconds for range end.
        fidelity : int, default 1
            Data resolution in minutes.

        Returns
        -------
        list[dict[str, Any]]
            List of price history points with 't' (timestamp) and 'p' (price).

        """
        PyCondition.valid_string(token_id, "token_id")

        # Convert milliseconds to seconds for the CLOB API
        start_time_s = start_time_ms // 1000
        end_time_s = end_time_ms // 1000

        params = {
            "market": token_id,
            "startTs": str(start_time_s),
            "endTs": str(end_time_s),
            "fidelity": str(fidelity),
        }
        response = await self._http_client.get(
            url="https://clob.polymarket.com/prices-history",
            params=params,
        )

        if response.status != 200:
            raise RuntimeError(
                f"HTTP request failed with status {response.status}: {response.body.decode('utf-8')}",
            )

        data = msgspec.json.decode(response.body)

        return data.get("history", [])

    def parse_orderbook_snapshots(
        self,
        snapshots: list[dict],
    ) -> list[OrderBookDeltas]:
        """
        Parse orderbook snapshots into OrderBookDeltas.

        Parameters
        ----------
        snapshots : list[dict]
            Raw orderbook snapshots from Polymarket CLOB API.

        Returns
        -------
        list[OrderBookDeltas]
            List of OrderBookDeltas for backtesting.

        """
        all_deltas: list[OrderBookDeltas] = []
        instrument_id = self._instrument.id
        make_price = self._instrument.make_price
        make_qty = self._instrument.make_qty

        # Skip zero-size entries as they represent no liquidity
        for snapshot in snapshots:
            ts_event = millis_to_nanos(int(snapshot["timestamp"]))

            deltas = [
                OrderBookDelta.clear(
                    instrument_id=instrument_id,
                    ts_event=ts_event,
                    ts_init=ts_event,
                    sequence=0,
                ),
            ]

            for bid in snapshot.get("bids", []):
                size_val = float(bid["size"])
                if size_val <= 0:
                    continue

                order = BookOrder(
                    side=OrderSide.BUY,
                    price=make_price(float(bid["price"])),
                    size=make_qty(size_val),
                    order_id=0,
                )
                deltas.append(
                    OrderBookDelta(
                        instrument_id=instrument_id,
                        action=BookAction.ADD,
                        order=order,
                        flags=0,
                        sequence=0,
                        ts_event=ts_event,
                        ts_init=ts_event,
                    ),
                )

            for ask in snapshot.get("asks", []):
                size_val = float(ask["size"])
                if size_val <= 0:
                    continue

                order = BookOrder(
                    side=OrderSide.SELL,
                    price=make_price(float(ask["price"])),
                    size=make_qty(size_val),
                    order_id=0,
                )
                deltas.append(
                    OrderBookDelta(
                        instrument_id=instrument_id,
                        action=BookAction.ADD,
                        order=order,
                        flags=0,
                        sequence=0,
                        ts_event=ts_event,
                        ts_init=ts_event,
                    ),
                )

            if deltas:
                all_deltas.append(OrderBookDeltas(instrument_id=instrument_id, deltas=deltas))

        return all_deltas

    def parse_price_history(
        self,
        history: list[dict],
    ) -> list[TradeTick]:
        """
        Parse price history into TradeTicks.

        Parameters
        ----------
        history : list[dict]
            Raw price history from CLOB API.

        Returns
        -------
        list[TradeTick]
            List of synthetic TradeTicks for backtesting.

        Notes
        -----
        Price history doesn't include actual trade information, so we synthesize
        trades from price points for demonstration purposes.

        """
        trades: list[TradeTick] = []

        for i, point in enumerate(history):
            timestamp = point["t"]  # Unix timestamp
            price_value = point["p"]

            ts_event = millis_to_nanos(int(timestamp * 1000))

            price = self._instrument.make_price(price_value)
            size = self._instrument.make_qty(1.0)

            # Determine aggressor side from price movement
            aggressor_side = AggressorSide.NO_AGGRESSOR
            if i > 0:
                prev_price = history[i - 1]["p"]
                if price_value > prev_price:
                    aggressor_side = AggressorSide.BUYER
                elif price_value < prev_price:
                    aggressor_side = AggressorSide.SELLER

            trade = TradeTick(
                instrument_id=self._instrument.id,
                price=price,
                size=size,
                aggressor_side=aggressor_side,
                trade_id=TradeId(str(i)),
                ts_event=ts_event,
                ts_init=ts_event,
            )
            trades.append(trade)

        return trades



import os
class FrozenParquetCache() :
    def __init__(self, conn_params):
        self.conn_params = conn_params
        cache_directory = ".frozen_parquet_cache"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)

        self._cache = {}
        for path in os.listdir(cache_directory):
            key = path.replace(".parquet", "")
            self._cache[key] = os.path.join(cache_directory, path)
    
    def _get_cache_key(self, instrument, start_time_ms, end_time_ms):
        return f"{instrument.id}_{start_time_ms}_{end_time_ms}"

    def _get_matching_cache_file(self, instrument, start_time_ms, end_time_ms):
        token_files = [f for f in self._cache.keys() if f.startswith(f"{instrument.id}_")]
        for token_file in token_files:
            _, file_start_str, file_end_str = token_file.split("_")
            file_start = int(file_start_str)
            file_end = int(file_end_str)
            if file_start <= start_time_ms and file_end >= end_time_ms:
                return self._cache[token_file]
        return None

    async def _clear_cache(self):
        cache_directory = ".frozen_parquet_cache"
        for path in os.listdir(cache_directory):
            os.remove(os.path.join(cache_directory, path))
        self._cache = {}  

    def fetch_orderbook_history_db(self, instrument, start_time_ms, end_time_ms):
        print(start_time_ms, end_time_ms)
        import io, tqdm
        if self.conn_params is None:
            raise ValueError("No QuestDB client provided for FzPolymarketDataLoader")

        QUERY = f"""SELECT ask_sizes, ask_prices, bid_sizes, bid_prices, timestamp, asset
        FROM book_history
        WHERE asset = '{get_polymarket_token_id(instrument.id)}'
        AND timestamp BETWEEN ({start_time_ms} * 1000)::timestamp AND ({end_time_ms} * 1000)::timestamp
        """

        url = f"http://{self.conn_params['host']}:{self.conn_params.get('port', 9000)}/exp"

        with requests.get(url = url, params={
            "query" : QUERY,
            "fmt" : "parquet"
        }, auth = (self.conn_params['user'], self.conn_params['password']), stream=True) as r:
            r.raise_for_status()

            total_size = int(r.headers.get('Content-Length', 0))
            buffer = io.BytesIO()

            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading - ") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        buffer.write(chunk)
                        pbar.update(len(chunk))

        df = pd.read_parquet(buffer)
        return df      

    async def _fetch_and_store(self, instrument, start_time_ms, end_time_ms):
        df = self.fetch_orderbook_history_db(instrument, start_time_ms, end_time_ms)
        if df.empty:
            return df
        cache_key = self._get_cache_key(instrument, start_time_ms, end_time_ms)
        cache_directory = ".frozen_parquet_cache"   
        cache_path = os.path.join(cache_directory, f"{cache_key}.parquet")
        df.to_parquet(cache_path)
        self._cache[cache_key] = cache_path
        return self._crop_dataframe(df, start_time_ms, end_time_ms)

    def _crop_dataframe(self, df, start_time_ms, end_time_ms):
        start = pd.to_datetime(start_time_ms, unit='ms', utc=True)
        end = pd.to_datetime(end_time_ms, unit='ms', utc=True)
        cropped_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        return cropped_df
    
    async def get(self, instrument , start_time_ms, end_time_ms):
        matching_file = self._get_matching_cache_file(instrument, start_time_ms, end_time_ms)
        if matching_file:
            print(f"Found matching cache file: {matching_file}")
            df = pd.read_parquet(matching_file)
            cropped_df = self._crop_dataframe(df, start_time_ms, end_time_ms)
            return cropped_df
        else :
            start_time_ns = instrument.activation_ns
            end_time_ns = instrument.expiration_ns

            start_date = pd.to_datetime(start_time_ns, unit='ns', utc=True)
            end_date = pd.to_datetime(end_time_ns, unit='ns', utc=True)
            print("No matching cache file found. Fetching from database...")
            print(f"Fetching data for market start: {start_date}, market end: {end_date}")
            return await self._fetch_and_store(instrument, end_time_ms=end_time_ns // 10**6, start_time_ms=start_time_ns // 10**6)
    

class FzPolymarketDataLoader(PolymarketDataLoader):
    def __init__(self, instrument, token_id = None, http_client = None):
        super().__init__(instrument, token_id, http_client)

    def set_conn_params(self, conn_params):
        self._conn_params = conn_params
        self._cache = FrozenParquetCache(conn_params)


    async def load_orderbook_snapshots(self, start, end):
        print(f"Loading orderbook snapshots for {self.instrument.id} from {start} to {end}")
        df = await self._cache.get(self.instrument, start.timestamp() * 1000, end.timestamp() * 1000)
        print("Data loaded, parsing snapshots...")
        if df.empty:
            return []
        
        all_deltas: list[OrderBookDeltas] = []
        instrument_id = self._instrument.id
        make_price = self._instrument.make_price
        make_qty = self._instrument.make_qty

        for row in df.itertuples():
            ts_event = millis_to_nanos(int(row.timestamp.value // 10**6))

            deltas = [
                OrderBookDelta.clear(
                    instrument_id=instrument_id,
                    ts_event=ts_event,
                    ts_init=ts_event,
                    sequence=0,
                ),
            ]

            bid_deltas = [OrderBookDelta(
                instrument_id=instrument_id,
                action=BookAction.ADD,
                order=BookOrder(
                    side=OrderSide.BUY,
                    price=make_price(float(bid_price)),
                    size=make_qty(float(bid_size)),
                    order_id=0,
                    ),
                    flags=0,
                    sequence=0,
                    ts_event=ts_event,
                    ts_init=ts_event,
                    ) for bid_price, bid_size in zip(row.bid_prices, row.bid_sizes) if float(bid_size) > 0]
            
            ask_deltas = [OrderBookDelta(
                instrument_id=instrument_id,
                action=BookAction.ADD,
                order=BookOrder(
                    side=OrderSide.SELL,
                    price=make_price(float(ask_price)),
                    size=make_qty(float(ask_size)),
                    order_id=0,
                    ),
                    flags=0,
                    sequence=0,
                    ts_event=ts_event,
                    ts_init=ts_event,
                    ) for ask_price, ask_size in zip(row.ask_prices, row.ask_sizes) if float(ask_size) > 0]
            
            deltas.extend(bid_deltas)
            deltas.extend(ask_deltas)

            if deltas:
                all_deltas.append(OrderBookDeltas(instrument_id=instrument_id, deltas=deltas))
        
        return all_deltas