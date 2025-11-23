"""
München Flughafen (MUC) - Echtzeit Flugdaten mit Metrik-Berechnung
====================================================================

Dieses Script:
1. Ruft Echtzeit-Flugdaten vom München Flughafen ab (ICAO: EDDM)
2. Berechnet Anzahl der Ankünfte und Abflüge in 30-Minuten-Intervallen
3. Berechnet die Metrik: 3 * (incoming_flights + outgoing_flights)

APIs zur Auswahl:
- OpenSky Network (kostenlos, keine Registration erforderlich)
- AviationStack (100 free requests/Monat, benötigt API Key)
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
import time
from typing import List, Dict, Tuple

class MunichAirportFlightData:
    """Klasse zum Abrufen und Verarbeiten von Flugdaten für München Airport"""

    # München Airport ICAO und IATA Codes
    MUNICH_ICAO = "EDDM"
    MUNICH_IATA = "MUC"

    def __init__(self, api_choice="opensky"):
        """
        Initialisiert die FlightData Klasse

        Args:
            api_choice: "opensky" oder "aviationstack"
        """
        self.api_choice = api_choice

    def get_flights_opensky(self, begin_time: int, end_time: int) -> Tuple[List, List]:
        """
        Holt Flugdaten von OpenSky Network API

        Args:
            begin_time: Unix timestamp (Startzeit)
            end_time: Unix timestamp (Endzeit)

        Returns:
            Tuple mit (arrivals, departures) Listen
        """
        base_url = "https://opensky-network.org/api"

        # Arrivals (Ankünfte in München)
        arrivals_url = f"{base_url}/flights/arrival"
        arrivals_params = {
            "airport": self.MUNICH_ICAO,
            "begin": begin_time,
            "end": end_time
        }

        # Departures (Abflüge von München)
        departures_url = f"{base_url}/flights/departure"
        departures_params = {
            "airport": self.MUNICH_ICAO,
            "begin": begin_time,
            "end": end_time
        }

        try:
            print(f"Rufe Ankünfte ab für {datetime.fromtimestamp(begin_time)} bis {datetime.fromtimestamp(end_time)}")
            arrivals_response = requests.get(arrivals_url, params=arrivals_params, timeout=10)

            # OpenSky hat Rate Limiting, kurz warten
            time.sleep(1)

            print(f"Rufe Abflüge ab...")
            departures_response = requests.get(departures_url, params=departures_params, timeout=10)

            arrivals = arrivals_response.json() if arrivals_response.status_code == 200 else []
            departures = departures_response.json() if departures_response.status_code == 200 else []

            print(f"✓ {len(arrivals)} Ankünfte gefunden")
            print(f"✓ {len(departures)} Abflüge gefunden")

            return arrivals, departures

        except Exception as e:
            print(f"Fehler beim Abrufen der Daten: {e}")
            return [], []

    # def get_flights_aviationstack(self, api_key: str) -> Tuple[List, List]:
    #     """
    #     Holt Flugdaten von AviationStack API (benötigt API Key)

    #     Args:
    #         api_key: Dein AviationStack API Key

    #     Returns:
    #         Tuple mit (arrivals, departures) Listen
    #     """
    #     base_url = "http://api.aviationstack.com/v1"

    #     # Arrivals
    #     arrivals_url = f"{base_url}/flights"
    #     arrivals_params = {
    #         "access_key": api_key,
    #         "arr_iata": self.MUNICH_IATA,
    #         "limit": 100
    #     }

    #     # Departures
    #     departures_params = {
    #         "access_key": api_key,
    #         "dep_iata": self.MUNICH_IATA,
    #         "limit": 100
    #     }

    #     try:
    #         arrivals_response = requests.get(arrivals_url, params=arrivals_params, timeout=10)
    #         departures_response = requests.get(arrivals_url, params=departures_params, timeout=10)

    #         arrivals = arrivals_response.json().get('data', []) if arrivals_response.status_code == 200 else []
    #         departures = departures_response.json().get('data', []) if departures_response.status_code == 200 else []

    #         return arrivals, departures

    #     except Exception as e:
    #         print(f"Fehler beim Abrufen der Daten: {e}")
    #         return [], []

    def group_flights_by_intervals(self, arrivals: List, departures: List, 
                                   interval_minutes: int = 30) -> pd.DataFrame:
        """
        Gruppiert Flüge nach Zeitintervallen und berechnet die Metrik

        Args:
            arrivals: Liste der Ankünfte
            departures: Liste der Abflüge
            interval_minutes: Intervall in Minuten (Standard: 30)

        Returns:
            DataFrame mit Zeitintervallen und berechneten Metriken
        """
        # Verarbeite Ankünfte
        arrival_times = []
        for flight in arrivals:
            if flight.get('lastSeen'):
                arrival_times.append(datetime.fromtimestamp(flight['lastSeen']))
            elif flight.get('firstSeen'):
                arrival_times.append(datetime.fromtimestamp(flight['firstSeen']))

        # Verarbeite Abflüge
        departure_times = []
        for flight in departures:
            if flight.get('firstSeen'):
                departure_times.append(datetime.fromtimestamp(flight['firstSeen']))

        # Erstelle DataFrame
        df_arrivals = pd.DataFrame({'time': arrival_times, 'type': 'arrival'})
        df_departures = pd.DataFrame({'time': departure_times, 'type': 'departure'})

        if df_arrivals.empty and df_departures.empty:
            print("⚠ Keine Flugdaten verfügbar")
            return pd.DataFrame()

        # Kombiniere beide DataFrames
        df_all = pd.concat([df_arrivals, df_departures], ignore_index=True)
        df_all = df_all.sort_values('time')

        # Erstelle Zeitintervalle
        min_time = df_all['time'].min().floor(f'{interval_minutes}min')
        max_time = df_all['time'].max().ceil(f'{interval_minutes}min')

        # Gruppiere nach Intervallen
        df_all['interval'] = df_all['time'].dt.floor(f'{interval_minutes}min')

        # Zähle Ankünfte und Abflüge pro Intervall
        grouped = df_all.groupby(['interval', 'type']).size().unstack(fill_value=0)

        # Stelle sicher, dass beide Spalten existieren
        if 'arrival' not in grouped.columns:
            grouped['arrival'] = 0
        if 'departure' not in grouped.columns:
            grouped['departure'] = 0

        # Berechne Metrik: 3 * (arrivals + departures)
        grouped['metric'] = 3 * (grouped['arrival'] + grouped['departure'])

        # Formatiere Ausgabe
        grouped = grouped.reset_index()
        grouped.columns = ['Time Interval', 'Arrivals', 'Departures', 'Metric (3*(A+D))']

        return grouped

    def calculate_settlement(self, df: pd.DataFrame) -> float:
        """
        Berechnet das Settlement wie im Beispiel

        Args:
            df: DataFrame mit Metriken

        Returns:
            Settlement Wert
        """
        if df.empty:
            return 0.0

        total_arrivals = df['Arrivals'].sum()
        total_departures = df['Departures'].sum()
        settlement = 3 * (total_arrivals + total_departures)

        return settlement


# ============================================================================
# HAUPTPROGRAMM - VERWENDUNGSBEISPIEL
# ============================================================================

def main():
    """Hauptfunktion mit Beispiel-Verwendung"""

    print("=" * 70)
    print("München Flughafen (MUC) - Echtzeit Flugdaten Analyse")
    print("=" * 70)
    print()

    # Initialisiere die Klasse
    flight_data = MunichAirportFlightData(api_choice="opensky")

    # Zeitraum definieren (letzte 3 Stunden als Beispiel)
    # Für Echtzeit-Daten: aktuelle Zeit bis jetzt
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)

    # Konvertiere zu Unix Timestamps
    begin_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())

    print(f"Zeitraum: {start_time.strftime('%Y-%m-%d %H:%M')} bis {end_time.strftime('%Y-%m-%d %H:%M')}")
    print()

    # Hole Flugdaten
    print("Rufe Flugdaten von OpenSky Network ab...")
    arrivals, departures = flight_data.get_flights_opensky(begin_timestamp, end_timestamp)

    if not arrivals and not departures:
        print("⚠ Keine Flugdaten verfügbar. Mögliche Gründe:")
        print("  - OpenSky API Rate Limit erreicht")
        print("  - Keine Flüge im gewählten Zeitraum")
        print("  - Netzwerkprobleme")
        return

    print()
    print("-" * 70)
    print()

    # Gruppiere nach 30-Minuten-Intervallen
    df_intervals = flight_data.group_flights_by_intervals(arrivals, departures, interval_minutes=30)

    if df_intervals.empty:
        print("Keine Daten zum Anzeigen")
        return

    # Zeige Ergebnisse
    print("Flugdaten nach 30-Minuten-Intervallen:")
    print()
    print(df_intervals.to_string(index=False))
    print()
    print("-" * 70)
    print()

    # Berechne Settlement
    settlement = flight_data.calculate_settlement(df_intervals)

    print("SETTLEMENT BERECHNUNG:")
    print(f"  Gesamt Ankünfte:   {df_intervals['Arrivals'].sum()}")
    print(f"  Gesamt Abflüge:    {df_intervals['Departures'].sum()}")
    print(f"  Settlement:        3 * ({df_intervals['Arrivals'].sum()} + {df_intervals['Departures'].sum()}) = {settlement}")
    print()
    print("=" * 70)

    # Optional: Speichere als CSV
    output_file = f"../data/munich_flights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df_intervals.to_csv(output_file, index=False)
    print(f"✓ Daten gespeichert in: {output_file}")


# ============================================================================
# ALTERNATIVE VERWENDUNG FÜR KONTINUIERLICHE ABFRAGEN
# ============================================================================

def continuous_monitoring(interval_seconds: int = 300):
    """
    Kontinuierliche Überwachung der Flugdaten

    Args:
        interval_seconds: Abfrageintervall in Sekunden (Standard: 300 = 5 Minuten)
    """
    flight_data = MunichAirportFlightData()

    print("Starte kontinuierliche Überwachung...")
    print(f"Abfrageintervall: {interval_seconds} Sekunden")
    print("Drücke Ctrl+C zum Beenden")
    print()

    try:
        while True:
            # Zeitfenster: letzte 30 Minuten
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=30)

            begin_ts = int(start_time.timestamp())
            end_ts = iynt(end_time.timestamp())

            arrivals, departures = flight_data.get_flights_opensky(begin_ts, end_ts)
            df = flight_data.group_flights_by_intervals(arrivals, departures, interval_minutes=30)

            if not df.empty:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(df.to_string(index=False))
                settlement = flight_data.calculate_settlement(df)
                print(f"Settlement (letzte 30 Min): {settlement}")

            # Warte bis zur nächsten Abfrage
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nÜberwachung beendet.")


if __name__ == "__main__":
    # Wähle Modus:

    # Modus 1: Einmalige Abfrage
    main()

    # Modus 2: Kontinuierliche Überwachung (auskommentiert)
    # continuous_monitoring(interval_seconds=300)