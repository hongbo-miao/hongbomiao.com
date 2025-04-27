import logging
import re
import time
import urllib.parse
from collections import OrderedDict
from typing import Any, TypedDict

import daft
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow as pa
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

logger = logging.getLogger(__name__)

pio.templates.default = "plotly_dark"
S3_BROWSER_URL = "https://s3-browser.hongbomiao.com"
TRINO_URL = "trino://trino_user@trino.hongbomiao.com/?source=trino-sqlalchemy"
DATA_SOURCES = ["motor", "battery"]
MAX_CACHE_SIZE = 10
MAX_RESULT_NUMBER = 5000
DEBOUNCE_TIME_MS = 300


class LRUCache(OrderedDict):
    def __init__(self, maxsize: int = MAX_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        super().__init__()

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        # Move accessed item to end to mark it as most recently used
        if key in self:
            self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        # If key exists, move it to the end for LRU tracking
        if key in self:
            self.move_to_end(key)
        # Add new item
        super().__setitem__(key, value)
        # Remove oldest item if cache is full
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
            logger.info(f"Cache full: Removed oldest item {oldest}")


# Replace cache dictionaries with LRU caches
cached_data: LRUCache = LRUCache(maxsize=MAX_CACHE_SIZE)
cached_signal_lists: LRUCache = LRUCache(maxsize=MAX_CACHE_SIZE // 2)
cached_filtered_signals: LRUCache = LRUCache(maxsize=100)


# Function to get event IDs from Trino based on data source
def get_event_ids(data_source: str) -> list[str]:
    sql_query: str = f"""
    select name
    from postgresql.public.{data_source}_events
    """  # noqa: S608
    event_df = daft.read_sql(sql_query, TRINO_URL)
    return event_df.to_arrow()["name"].to_pylist()


# Modified function to load multiple signals
def load_data(  # noqa: C901, PLR0912
    data_source: str,
    event_id: str,
    selected_signals: str | list[str] | None = None,
) -> tuple[dict[str, pa.Table] | None, list[str]]:
    logger.info(f"Loading data for {data_source} event {event_id}...")
    # Create a unique cache key
    cache_key: str = f"{data_source}_{event_id}"

    # Check if we have the signal list cached
    if selected_signals is None and cache_key in cached_signal_lists:
        return None, cached_signal_lists[cache_key]

    # If we already have the column names cached
    if cache_key in cached_data and cached_data[cache_key][1] is not None:
        all_signals: list[str] = cached_data[cache_key][1]
        # If we just need the list of signals (without loading data)
        if selected_signals is None:
            return None, all_signals
    else:
        # Need to fetch column names from the parquet file
        try:
            # Load the metadata to get column names
            df_daft = daft.read_parquet(
                f"{S3_BROWSER_URL}/{data_source}_parquet/{event_id}.parquet",
            )
            all_signals = df_daft.column_names

            # Store the column names in cache
            cached_data[cache_key] = (None, all_signals)
            cached_signal_lists[cache_key] = all_signals  # Cache the signal list

            # If we just need the list of signals (without loading data)
            if selected_signals is None:
                return None, all_signals
        except Exception as e:  # noqa: BLE001
            logger.info(f"Error getting column names: {e = }")

            # Fallback to empty list if we can't get column names
            all_signals = []
            cached_data[cache_key] = (None, all_signals)
            cached_signal_lists[cache_key] = all_signals  # Cache the empty list
            return None, all_signals

    # Handle single signal passed as string
    if isinstance(selected_signals, str):
        selected_signals = [selected_signals]

    # If no signals were selected, return None
    if not selected_signals:
        return None, all_signals

    # Try to load from cache first
    result_tables: dict[str, pa.Table] = {}
    signals_to_load: list[str] = []

    # Check which signals we need to load vs which are in cache
    for signal in selected_signals:
        signal_cache_key: str = f"{cache_key}_{signal}"
        if signal_cache_key in cached_data:
            logger.info(
                f"Using cached data for {data_source} event {event_id}, signal: {signal}",
            )
            result_tables[signal] = cached_data[signal_cache_key]
        else:
            signals_to_load.append(signal)

    # If we have signals to load from parquet
    if signals_to_load:
        # Load only the required columns (_time and the selected signals)
        columns_to_load: list[str] = ["_time", *signals_to_load]
        df_daft = daft.read_parquet(
            f"{S3_BROWSER_URL}/{data_source}_parquet/{event_id}.parquet",
        )
        df_daft = df_daft.select(*columns_to_load)

        # Convert to Arrow Table
        full_table: pa.Table = df_daft.to_arrow()

        # Cache and store each signal separately
        for signal in signals_to_load:
            # Extract just the time and this signal
            signal_table: pa.Table = full_table.select(["_time", signal])
            signal_cache_key = f"{cache_key}_{signal}"
            cached_data[signal_cache_key] = signal_table
            result_tables[signal] = signal_table
            logger.info(
                f"Data loaded and cached for {data_source} event {event_id}, signal: {signal}",
            )
    return result_tables, all_signals


# Create a dynamic resampling function that takes visible range into account
def dynamic_downsample(
    table: pa.Table,
    x_column: str,
    y_column: str,
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    max_points: int = 10000,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    # Convert time column to timestamps if it's not already
    x_array: pd.DatetimeIndex = pd.to_datetime(table[x_column].to_numpy(), unit="ns")
    y_array: np.ndarray = table[y_column].to_numpy()

    # Filter by range if specified
    if x_range is not None:
        mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
        x_array = x_array[mask]
        y_array = y_array[mask]

    # If we have fewer points than max_points, return all points
    if len(x_array) <= max_points:
        return x_array, y_array

    # Otherwise, downsample
    step = len(x_array) // max_points
    return x_array[::step], y_array[::step]


# Build signal search index (call once when signals are loaded)
def build_signal_index(signals: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for signal in signals:
        # Skip special signals
        if signal == "_time":
            continue

        # Add to term index
        signal_lower: str = signal.lower()

        # Add whole term
        if signal_lower not in index:
            index[signal_lower] = []
        index[signal_lower].append(signal)

        # Add each word
        for word in re.findall(r"[a-z0-9_]+", signal_lower):
            if word not in index:
                index[word] = []
            if signal not in index[word]:
                index[word].append(signal)
    return index


# Improved search function with advanced techniques
def advanced_search(  # noqa: C901, PLR0912
    signal_list: list[str],
    search_term: str,
    max_results: int = MAX_RESULT_NUMBER,
) -> list[str]:
    if not search_term or search_term.strip() == "":
        return []

    search_term = search_term.strip().lower()

    # Create a cache key for filtered signals
    cache_key: str = f"filter_{search_term}"

    # Check if we have cached results for this search term
    if cache_key in cached_filtered_signals:
        logger.info(f"Using cached results for search term: '{search_term}'")
        return cached_filtered_signals[cache_key]

    # Check if we have an index for these signals
    index_key: str = f"index_{len(signal_list)}"
    if index_key not in cached_filtered_signals:
        # Build index if we don't have one
        logger.info(f"Building search index for {len(signal_list)} signals")
        signal_index: dict[str, list[str]] = build_signal_index(signal_list)
        cached_filtered_signals[index_key] = signal_index
    else:
        signal_index = cached_filtered_signals[index_key]

    # Split the search term into words for multi-term search
    search_words: list[str] = re.findall(r"[a-z0-9_]+", search_term)
    if not search_words:
        return []

    # Different matching categories (ordered by relevance)
    exact_matches: list[str] = []
    prefix_matches: list[str] = []
    all_words_matches: list[str] = []
    any_word_matches: list[str] = []
    substring_matches: list[str] = []

    # Set to track what we've already found to avoid duplicates
    found_signals: set[str] = set()

    # First try exact match on the full search term
    if search_term in signal_index:
        for signal in signal_index[search_term]:
            if signal not in found_signals:
                exact_matches.append(signal)
                found_signals.add(signal)

    # Then try prefix matches (for each word in signals)
    for signal in signal_list:
        if signal == "_time" or signal in found_signals:
            continue

        signal_lower: str = signal.lower()
        words: list[str] = re.findall(r"[a-z0-9_]+", signal_lower)
        # Check if any word in the signal starts with the search term
        for word in words:
            if word.startswith(search_term):
                prefix_matches.append(signal)
                found_signals.add(signal)
                break

    # All words match - every search word must be in the signal
    if len(search_words) > 1:  # Only relevant for multi-word searches
        for signal in signal_list:
            if signal == "_time" or signal in found_signals:
                continue

            signal_lower = signal.lower()
            if all(word in signal_lower for word in search_words):
                all_words_matches.append(signal)
                found_signals.add(signal)

    # Check for full substring match (the entire search term appears somewhere)
    for signal in signal_list:
        if signal == "_time" or signal in found_signals:
            continue

        signal_lower = signal.lower()
        if search_term in signal_lower:
            substring_matches.append(signal)
            found_signals.add(signal)

    # Any word match - at least one search word must be in the signal
    for word in search_words:
        if word in signal_index:
            for signal in signal_index[word]:
                if signal not in found_signals:
                    any_word_matches.append(signal)
                    found_signals.add(signal)

    # Combine results in order of relevance
    results: list[str] = (
        exact_matches
        + prefix_matches
        + all_words_matches
        + substring_matches
        + any_word_matches
    )

    # Limit to max_results
    limited_results: list[str] = results[:max_results]
    # Cache and return results
    cached_filtered_signals[cache_key] = limited_results

    # Log search statistics
    found_count: int = len(limited_results)
    total_found: int = len(results)
    logger.info(f"Search '{search_term}': Found {found_count}/{total_found} matches")
    logger.info(
        f"  Exact: {len(exact_matches)}, Prefix: {len(prefix_matches)}, "
        f"All words: {len(all_words_matches)}, Substring: {len(substring_matches)}, "
        f"Any word: {len(any_word_matches)}",
    )

    return limited_results


class StoredEventData(TypedDict):
    data_source: str
    event_id: str
    signals: list[str]


class UrlParams(TypedDict, total=False):
    data_source: str | None
    event_id: str | None
    signal: list[str] | str


class InitStatus(TypedDict):
    is_initialized: bool


# Create Dash app
app = Dash(__name__, suppress_callback_exceptions=True, title="Parquet Visualizer")

# Define layout with data source dropdown and event_id dropdown
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),  # URL tracking component
        html.H1("Parquet Visualizer"),
        html.Div(
            [
                html.Label("Select data source:"),
                dcc.Dropdown(
                    id="data-source-dropdown",
                    options=[{"label": ds, "value": ds} for ds in DATA_SOURCES],
                    value=None,
                    placeholder="Select a data source",
                ),
            ],
            style={"width": "100%", "margin": "10px 0"},
        ),
        html.Div(
            [
                html.Label("Select event ID:"),
                dcc.Dropdown(
                    id="event-id-dropdown",
                    options=[],  # Will be populated based on data source selection
                    value=None,
                    placeholder="Select a data source first",
                    disabled=True,
                ),
            ],
            style={"width": "100%", "margin": "10px 0"},
        ),
        html.Div(
            id="loading-status",
            children="Select a data source and event ID to load data",
        ),
        html.Div(
            id="visualization-container",
            children=[
                html.Div(
                    [
                        html.Label("Search and select signals:"),
                        # Input with debounce (only fires callback after user stops typing)
                        dcc.Input(
                            id="signal-search",
                            type="text",
                            placeholder="Type to search signals...",
                            style={"width": "100%", "margin": "10px 0"},
                            # Add debounce to only fire callback after user stops typing
                            # Don't use debounce directly on Input as we want immediate UI feedback
                        ),
                        # Add loading status indicator for search results
                        html.Div(
                            id="dropdown-loading-indicator",
                            style={"margin": "5px 0", "color": "#aaa"},
                        ),
                        dcc.Dropdown(
                            id="signal-dropdown",
                            placeholder="Type in the search box above to find signals",
                            multi=True,  # Enable multiple selection
                            style={"width": "100%", "margin": "10px 0"},
                            optionHeight=30,  # Slightly smaller option height for more compact display
                            maxHeight=400,  # Ensure dropdown doesn't get too large
                        ),
                    ],
                    style={"width": "100%", "margin": "10px 0"},
                ),
                # Add loading wrapper around the graph
                dcc.Loading(
                    id="graph-loading",
                    type="circle",
                    children=[
                        dcc.Graph(id="time-series-graph"),
                    ],
                ),
                html.Div(id="sample-info", style={"margin": "10px 0"}),
            ],
        ),
        # Store component to keep track of loaded data and signal structure
        dcc.Store(id="stored-event-data", data=None),
        # Store URL parameters for initialization
        dcc.Store(id="url-parameters", data={}),
        # Flag to track initialization
        dcc.Store(id="initialization-status", data={"is_initialized": False}),
        # Store for debounced search term
        dcc.Store(id="debounced-search", data=None),
        # Interval for debouncing
        dcc.Interval(
            id="debounce-interval",
            interval=DEBOUNCE_TIME_MS,
            n_intervals=0,
            disabled=True,
        ),
        # Store timestamp of last search input
        dcc.Store(id="last-search-time", data=0),
    ],
)


# Parse URL parameters when the app loads
@app.callback(
    Output("url-parameters", "data"),
    Output("initialization-status", "data"),
    Input("url", "href"),
    State("initialization-status", "data"),
)
def parse_url_parameters(
    href: str | None,
    initialization_status: InitStatus,
) -> tuple[UrlParams, InitStatus]:
    if initialization_status.get("is_initialized", False):
        # Avoid re-initializing if already done
        raise PreventUpdate
    if not href:
        return {}, {"is_initialized": True}
    try:
        # Parse the URL parameters
        parsed_url = urllib.parse.urlparse(href)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        # Extract the parameters
        params: UrlParams = {
            "data_source": query_params.get("data_source", [None])[0],
            "event_id": query_params.get("event_id", [None])[0],
            "signal": query_params.get("signal", []),  # Get signals as a list
        }
    except Exception as e:  # noqa: BLE001
        logger.info(f"Error parsing URL parameters: {e = }")
        return {}, {"is_initialized": True}
    else:
        return params, {"is_initialized": True}


# Initialize the data source dropdown from URL parameters
@app.callback(
    Output("data-source-dropdown", "value"),
    Input("url-parameters", "data"),
)
def initialize_data_source(url_params: UrlParams) -> str | None:
    data_source = url_params.get("data_source")
    if data_source in DATA_SOURCES:
        return data_source
    return None


# Callback to update URL when selections change
@app.callback(
    Output("url", "search"),
    Input("data-source-dropdown", "value"),
    Input("event-id-dropdown", "value"),
    Input("signal-dropdown", "value"),
    State("initialization-status", "data"),
)
def update_url(
    data_source: str | None,
    event_id: str | None,
    signals: list[str] | None,
    initialization_status: InitStatus,
) -> str:
    if not initialization_status.get("is_initialized", False):
        raise PreventUpdate
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    # Build the query parameters
    params: dict[str, str | list[str]] = {}
    if data_source:
        params["data_source"] = data_source
    if event_id:
        params["event_id"] = event_id
    # Handle multiple signals
    if signals:
        if isinstance(signals, list):
            # Create a list of signal parameters
            signal_params: list[tuple[str, str]] = [
                ("signal", signal) for signal in signals
            ]
            # Return the constructed URL
            query_parts: list[str] = []
            if data_source:
                query_parts.append(f"data_source={urllib.parse.quote(data_source)}")
            if event_id:
                query_parts.append(f"event_id={urllib.parse.quote(event_id)}")
            for param, value in signal_params:
                query_parts.append(f"{param}={urllib.parse.quote(value)}")
            return "?" + "&".join(query_parts)
        # Single signal (shouldn't happen with multi=True)
        params["signal"] = signals
    # Create the URL query string
    query_string: str = urllib.parse.urlencode(
        params,
        doseq=True,
    )  # doseq=True handles lists
    return f"?{query_string}" if query_string else ""


# Callback to update event ID dropdown when data source changes
@app.callback(
    Output("event-id-dropdown", "options"),
    Output("event-id-dropdown", "disabled"),
    Output("loading-status", "children"),
    Input("data-source-dropdown", "value"),
)
def update_event_ids(
    data_source: str | None,
) -> tuple[list[dict[str, str]], bool, str]:
    if not data_source:
        return [], True, "Select a data source first"
    try:
        event_ids: list[str] = get_event_ids(data_source)
        # Sort event IDs from Z to A
        event_ids.sort(reverse=True)
        options: list[dict[str, str]] = [
            {"label": event_id, "value": event_id} for event_id in event_ids
        ]
    except Exception as e:  # noqa: BLE001
        return [], True, f"Error loading event IDs for {data_source}: {e = }"
    else:
        return (
            options,
            False,
            f"Selected data source: {data_source}. Now select an event ID.",
        )


# Callback to set the event ID from URL parameters after options are loaded
@app.callback(
    Output("event-id-dropdown", "value"),
    Input("event-id-dropdown", "options"),
    State("url-parameters", "data"),
    State("event-id-dropdown", "value"),
)
def set_event_id_from_url(
    event_options: list[dict[str, str]],
    url_params: UrlParams,
    current_value: str | None,
) -> str | None:
    # If there's already a value selected, don't override it
    if current_value:
        return current_value
    event_id = url_params.get("event_id")
    if event_id and event_options:
        # Check if the event_id from URL is in the available options
        event_values: list[str] = [opt["value"] for opt in event_options]
        if event_id in event_values:
            return event_id
    return None


# Callback to load data when event ID changes
@app.callback(
    Output("loading-status", "children", allow_duplicate=True),
    Output("visualization-container", "style"),
    Output("stored-event-data", "data"),
    Input("event-id-dropdown", "value"),
    State("data-source-dropdown", "value"),
    prevent_initial_call=True,
)
def load_data_on_event_change(
    selected_event_id: str | None,
    data_source: str | None,
) -> tuple[str, dict[str, str], StoredEventData | None]:
    if not selected_event_id or not data_source:
        return (
            "Select a data source and event ID to load data",
            {"display": "none"},
            None,
        )
    try:
        # Just get the list of signals without loading actual data
        _, all_signals = load_data(data_source, selected_event_id)
        # Store information about the loaded event
        stored_data: StoredEventData = {
            "data_source": data_source,
            "event_id": selected_event_id,
            "signals": all_signals,
        }
        # Count signals
        signal_count: int = len(all_signals)
    except Exception as e:  # noqa: BLE001
        return f"Error loading data: {e = }", {"display": "none"}, None
    else:
        return (
            f"Data loaded for {data_source} event {selected_event_id}. Total signals available: {signal_count}.",
            {"display": "block"},
            stored_data,
        )


# Callback to initialize signal search field from URL
@app.callback(
    Output("signal-search", "value"),
    Input("stored-event-data", "data"),
    State("url-parameters", "data"),
)
def initialize_signal_search(
    stored_data: StoredEventData | None,
    url_params: UrlParams,
) -> str | None:
    if not stored_data:
        return None
    signals = url_params.get("signal", [])
    # For multiple signals in URL, don't set a search term
    # This allows us to display exactly the signals specified in URL
    if signals and isinstance(signals, list) and len(signals) > 0:
        return None
    # For single signal, use it as search term
    if signals and isinstance(signals, str):
        return signals
    return None


# Debounce implementation
@app.callback(
    Output("debounce-interval", "disabled"),
    Output("last-search-time", "data"),
    Input("signal-search", "value"),
)
def handle_search_input(search_term: str | None) -> tuple[bool, float]:  # noqa: ARG001
    # Enable the interval and record the current time
    current_time: float = time.time() * 1000  # Convert to milliseconds
    return False, current_time


@app.callback(
    Output("debounced-search", "data"),
    Input("debounce-interval", "n_intervals"),
    Input("last-search-time", "data"),
    State("signal-search", "value"),
    State("debounced-search", "data"),
)
def update_debounced_search(
    n_intervals: int,  # noqa: ARG001
    last_time: float,
    current_search: str | None,
    previous_search: str | None,
) -> str | None:
    # Check if enough time has passed since the last input
    current_time: float = time.time() * 1000
    if (
        current_time - last_time >= DEBOUNCE_TIME_MS
        and current_search != previous_search
    ):
        return current_search
    raise PreventUpdate


# Callback to update dropdown loading indicator
@app.callback(
    Output("dropdown-loading-indicator", "children"),
    Input("signal-search", "value"),
    Input("signal-dropdown", "options"),
)
def update_dropdown_status(
    search_term: str | None,
    options: list[dict[str, str]] | None,
) -> str:
    if not search_term:
        return ""

    num_options: int = len(options) if options else 0
    if num_options == 0:
        if search_term.strip() == "":
            return ""
        return "No matches found. Try a different search term."
    if num_options >= MAX_RESULT_NUMBER:
        return f"Found {num_options}+ matches for '{search_term}'. Showing top {num_options} results."
    return f"Found {num_options} matches for '{search_term}'."


# Callback to update signal dropdown options based on search input
@app.callback(
    Output("signal-dropdown", "options"),
    Output("signal-dropdown", "value", allow_duplicate=True),
    Input("debounced-search", "data"),  # Use debounced search term
    Input("stored-event-data", "data"),
    State("signal-dropdown", "value"),
    State("url-parameters", "data"),
    prevent_initial_call=True,
)
def update_signal_dropdown(  # noqa: C901
    search_term: str | None,
    stored_data: StoredEventData | None,
    current_signal_values: list[str] | None,
    url_params: UrlParams,
) -> tuple[list[dict[str, str]], list[str]]:
    if not stored_data:
        return [], []

    ctx = callback_context

    triggered_id: str | None = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    )
    try:
        all_signals: list[str] = stored_data.get("signals", [])
        # Filter signals based on search term if provided
        if search_term:
            filtered_signals: list[str] = advanced_search(
                all_signals,
                search_term,
            )
        # If no search term, show specified signals or none
        elif current_signal_values:
            # If we already have selections, just keep those
            filtered_signals = [s for s in current_signal_values if s in all_signals]
        else:
            # Default to empty when no search term and no selections
            filtered_signals = []

        # Create dropdown options
        options: list[dict[str, str]] = [
            {"label": signal, "value": signal} for signal in filtered_signals
        ]
        logger.info(
            f"Created {len(options)} dropdown options. First 3: {options[:3] if options else 'none'}",
        )

        # If the trigger was a search term change, we should consider selecting the first option
        values: list[str] = []
        if (
            triggered_id == "debounced-search"
            and search_term
            and options
            and not current_signal_values
        ):
            values = [options[0]["value"]]
            logger.info(f"Auto-selecting first option: {values[0]}")
            return options, values

        # Get signals from URL if available
        signals_from_url = url_params.get("signal", [])

        # If there are signals in the URL, use them
        if signals_from_url:
            # Filter to only include signals that are in our available options
            valid_signals: list[str] = (
                [s for s in signals_from_url if s in all_signals]
                if isinstance(signals_from_url, list)
                else []
            )
            if valid_signals:
                return [
                    {"label": signal, "value": signal} for signal in valid_signals
                ], valid_signals

        # If we already have selections, keep them
        if current_signal_values:
            if isinstance(current_signal_values, str):
                current_signal_values = [current_signal_values]
            # Keep only values that are valid signals
            values = [s for s in current_signal_values if s in all_signals]
    except Exception:
        logger.exception("Error updating signal dropdown")
        return [], []
    else:
        return options, values


# Callback to update graph based on selections and zoom level
@app.callback(
    Output("time-series-graph", "figure"),
    Output("sample-info", "children"),
    Input("signal-dropdown", "value"),
    Input("time-series-graph", "relayoutData"),
    State("stored-event-data", "data"),
    prevent_initial_call=True,
)
def update_graph(  # noqa: C901
    signals: str | list[str] | None,
    relayout_data: dict[str, Any] | None,
    stored_data: StoredEventData | None,
) -> tuple[go.Figure, str]:
    if not signals or not stored_data:
        return {}, "No signals selected or no data loaded"

    # Convert to list if it's a single value
    if isinstance(signals, str):
        signals = [signals]
    if not signals:  # If signals is empty list
        return {}, "No signals selected"
    data_source: str = stored_data["data_source"]
    event_id: str = stored_data["event_id"]

    # Load data for all selected signals
    signal_tables: dict[str, pa.Table] | None
    signal_tables, _ = load_data(data_source, event_id, signals)
    if not signal_tables:
        return {}, "Failed to load signals data"

    # Default range is the full dataset
    x_range: tuple[pd.Timestamp, pd.Timestamp] | None = None

    # Get timestamps from the first signal (they should all have the same time base)
    first_signal: str = next(iter(signal_tables.keys()))
    timestamps: pd.DatetimeIndex = pd.to_datetime(
        signal_tables[first_signal]["_time"].to_numpy(),
        unit="ns",
    )
    # Check if we have a zoom event with new x-axis bounds
    if relayout_data and (
        "xaxis.range" in relayout_data or "xaxis.range[0]" in relayout_data
    ):
        if "xaxis.range" in relayout_data:
            x_range = (
                pd.Timestamp(relayout_data["xaxis.range"][0]),
                pd.Timestamp(relayout_data["xaxis.range"][1]),
            )
        else:
            x_range = (
                pd.Timestamp(relayout_data["xaxis.range[0]"]),
                pd.Timestamp(relayout_data["xaxis.range[1]"]),
            )
    # Create the figure
    fig = go.Figure()
    # Add trace for each signal
    for signal in signals:
        if signal in signal_tables:
            # Apply dynamic downsampling
            x_downsampled, y_downsampled = dynamic_downsample(
                signal_tables[signal],
                "_time",
                signal,
                x_range,
            )
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=x_downsampled,
                    y=y_downsampled,
                    mode="lines",
                    name=signal,
                ),
            )
    # Update layout
    fig.update_layout(
        title=f"{len(signals)} signal(s) selected",
        xaxis={"title": "Time"},
        yaxis={"title": "Value"},
        height=700,
        template="plotly_dark",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
        },
    )
    # If we have a specific range, preserve it
    if x_range:
        fig.update_layout(xaxis_range=[x_range[0], x_range[1]])
    # Display information about the visible data
    info_text: str = f"Displaying {len(signals)} signal(s)"
    if x_range:
        visible_start: str = x_range[0].strftime("%Y-%m-%d %H:%M:%S")
        visible_end: str = x_range[1].strftime("%Y-%m-%d %H:%M:%S")
        info_text += f" from {visible_start} to {visible_end}"

        # Calculate what percentage of the full dataset is being shown
        full_range: float = (timestamps.max() - timestamps.min()).total_seconds()
        visible_range: float = (x_range[1] - x_range[0]).total_seconds()
        percentage: float = (visible_range / full_range) * 100
        info_text += f" ({percentage:.2f}% of the full time range)"
    return fig, info_text


server = app.server
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting Dash server...")
    app.run(debug=False, port=8050)
