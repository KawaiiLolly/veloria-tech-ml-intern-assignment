"""
    scraper.py — Cricket Match Data Collector
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from fallback_data import FALLBACK_DATA

OUTPUT_FILE = "match_data.csv"
REQUEST_DELAY = 1.5          
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None


def parse_howstat_results(soup: BeautifulSoup, team_name: str) -> list[dict]:
    matches = []
    try:
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header row
                cols = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cols) >= 5:
                    matches.append({
                        "match_date": cols[0] if cols[0] else "N/A",
                        "team1": team_name,
                        "team2": cols[1] if len(cols) > 1 else "N/A",
                        "venue": cols[2] if len(cols) > 2 else "N/A",
                        "winner": cols[3] if len(cols) > 3 else "N/A",
                        "top_scorer": "N/A",
                        "top_score": 0,
                        "home_team": "N/A",
                    })
                    if len(matches) == 10:
                        break
            if len(matches) == 10:
                break
    except Exception as e:
        print(f"Parsing failed: {e}")
    return matches


def try_live_scrape() -> list[dict] | None:
    """
    Attempt live scraping of HowStat for India + Australia.
    Returns combined list if successful, None otherwise.
    """
    base = "http://www.howstat.com/cricket/Statistics/Matches/MatchList_ODI.asp"
    teams = {
        "India": "IND",
        "Australia": "AUS",
    }
    all_matches = []
    success = True

    for team_name, code in teams.items():
        url = f"{base}?TeamCode={code}"
        soup = fetch_page(url)
        if soup is None:
            success = False
            break
        results = parse_howstat_results(soup, team_name)
        if not results:
            print(f"No results parsed for {team_name}")
            success = False
            break
        all_matches.extend(results[:10])

    return all_matches if success and len(all_matches) >= 10 else None

def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert list of match dicts to a clean, typed DataFrame."""
    df = pd.DataFrame(records)

    columns = [
        "match_date", "team1", "team2", "venue",
        "winner", "top_scorer", "top_score", "home_team",
    ]
    df = df[columns]

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["top_score"] = pd.to_numeric(df["top_score"], errors="coerce").fillna(0).astype(int)

    critical = ["team1", "team2", "winner"]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} incomplete rows.")
    df = df.reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("  Cricket Match Data Collector — scraper.py")
    print("=" * 60)

    print("Attempting live scrape from HowStat…")
    live_data = try_live_scrape()

    if live_data:
        print(f"Live scrape succeeded — {len(live_data)} records collected.")
        records = live_data
    else:
        print(
            "Live scrape unavailable (site may be blocking automated requests). "
            "Using curated fallback dataset of real match results."
        )
        records = FALLBACK_DATA

    df = build_dataframe(records)
    print(f"Dataset built: {len(df)} rows × {len(df.columns)} columns")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}")

    print("\n── Dataset Preview (first 5 rows) ──────────────────────")
    print(df.head().to_string(index=False))
    print(f"\nTotal records : {len(df)}")
    print(f"Teams covered : {sorted(df['team1'].unique().tolist())}")
    print(f"Date range    : {df['match_date'].min().date()} → {df['match_date'].max().date()}")
    print(f"Output file   : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
