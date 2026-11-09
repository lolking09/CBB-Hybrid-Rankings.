"""Automatic college basketball hybrid rankings, standard library only."""
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SEASON = 2027  # The 2026-27 college basketball season
HOME_ADVANTAGE = 3.0
PREDICTIVE_WEIGHT = 0.50
PRIOR_GAMES = 5.0
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'
API = 'https://api.collegebasketballdata.com'


def request_json(path, params):
    key = os.environ.get('CBBD_API_KEY')
    if not key:
        raise RuntimeError('CBBD_API_KEY secret is missing')
    url = API + path + '?' + urlencode(params)
    req = Request(url, headers={'Authorization': 'Bearer ' + key,
                                'Accept': 'application/json',
                                'User-Agent': 'CBB-Hybrid-Rankings/1.0'})
    with urlopen(req, timeout=45) as response:
        return json.load(response)


def load_cache(path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def update_data():
    DATA.mkdir(exist_ok=True)
    game_path = DATA / 'games_2027.json'
    team_path = DATA / 'teams_2027.json'
    existing = load_cache(game_path, [])
    today = datetime.now(timezone.utc)
    # Backfill the full season on first run, using month-sized date windows.
    if not existing:
        start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    else:
        start = today - timedelta(days=5)
    by_id = {str(game['id']): game for game in existing if game.get('id') is not None}
    while start < today:
        end = min(start + timedelta(days=28), today)
        rows = request_json('/games', {
            'season': SEASON,
            'startDateRange': start.isoformat(),
            'endDateRange': end.isoformat(),
        })
        if not isinstance(rows, list):
            raise RuntimeError('Unexpected /games response; refusing to overwrite cache')
        if len(rows) >= 3000:
            raise RuntimeError('API result may be truncated; shorten query window')
        for game in rows:
            if isinstance(game, dict) and game.get('id') is not None:
                by_id[str(game['id'])] = game
        start = end
    if not team_path.exists():
        teams = request_json('/teams', {'season': SEASON})
        if not isinstance(teams, list) or not teams:
            raise RuntimeError('Unexpected /teams response')
        save_json(team_path, teams)
    games = sorted(by_id.values(), key=lambda g: (str(g.get('startDate', '')), str(g.get('id', ''))))
    if not games:
        raise RuntimeError('No games returned; rankings not generated')
    save_json(game_path, games)
    return games, load_cache(team_path, [])


def standardize(games, team_rows):
    # /teams?season supplies the season's Division I team list.
    names = {int(t['id']): t.get('school') or t.get('displayName') or str(t['id'])
             for t in team_rows if t.get('id') is not None}
    if not names:
        raise RuntimeError('No Division I teams in team list')
    good = []
    for game in games:
        try:
            h = int(game['homeTeamId'])
            a = int(game['awayTeamId'])
            hs = game['homePoints']
            aw = game['awayPoints']
            if h not in names or a not in names or hs is None or aw is None:
                continue
            hs, aw = int(hs), int(aw)
            if hs == aw or hs < 0 or aw < 0:
                continue
            # An unfinished game should not be counted when its score exists.
            status = str(game.get('status') or '').lower()
            if status not in ('completed', 'final', 'complete'):
                continue
            good.append((h, a, hs, aw, bool(game.get('neutralSite'))))
        except (KeyError, ValueError, TypeError):
            continue
    return names, good


def normalize(values):
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values.values()) / len(values))
    return {t: 50.0 if sd == 0 else max(0, min(100, 50 + 15 * (v - mean) / sd))
            for t, v in values.items()}


def prob(edge):
    return 1 / (1 + math.exp(-max(-500, min(500, edge)) / 11))


def calculate(names, games):
    played = defaultdict(int)
    wins = defaultdict(int)
    for home, away, hs, aw, neutral in games:
        played[home] += 1
        played[away] += 1
        wins[home if hs > aw else away] += 1
    active = sorted(t for t in names if played[t] > 0)
    if not active:
        raise RuntimeError('No final D-I-vs-D-I games found; check API data/status')
    power = {t: 0.0 for t in active}
    for _ in range(200):
        totals = defaultdict(float)
        for h, a, hs, aw, neutral in games:
            venue = 0 if neutral else HOME_ADVANTAGE
            totals[h] += hs - aw - venue + power[a]
            totals[a] += aw - hs + venue + power[h]
        updated = {t: totals[t] / (played[t] + PRIOR_GAMES) for t in active}
        center = sum(updated.values()) / len(updated)
        power = {t: 0.5 * power[t] + 0.5 * (updated[t] - center) for t in active}
    wae = defaultdict(float)
    sos = defaultdict(float)
    for h, a, hs, aw, neutral in games:
        venue = 0 if neutral else HOME_ADVANTAGE
        wae[h] += (hs > aw) - prob(venue - power[a])
        wae[a] += (aw > hs) - prob(-venue - power[h])
        sos[h] += power[a]
        sos[a] += power[h]
    pscore = normalize(power)
    rscore = normalize({t: wae[t] / math.sqrt(played[t] + PRIOR_GAMES) for t in active})
    result = []
    for t in active:
        result.append({
            'team': names[t], 'team_id': t, 'wins': wins[t],
            'losses': played[t] - wins[t],
            'predictive': round(pscore[t], 2),
            'resume': round(rscore[t], 2),
            'hybrid': round(PREDICTIVE_WEIGHT * pscore[t] + (1 - PREDICTIVE_WEIGHT) * rscore[t], 2),
            'power_margin': round(power[t], 2),
            'wins_above_expected': round(wae[t], 3),
            'schedule_strength': round(sos[t] / played[t], 2),
        })
    result.sort(key=lambda r: (-r['hybrid'], -r['predictive'], r['team']))
    for rank, r in enumerate(result, 1):
        r['rank'] = rank
    return result


def write_outputs(rows, number_of_games):
    OUTPUT.mkdir(exist_ok=True)
    fields = ['rank', 'team', 'wins', 'losses', 'hybrid', 'predictive', 'resume',
              'power_margin', 'wins_above_expected', 'schedule_strength', 'team_id']
    for filename, selected in [('rankings.csv', rows), ('top25.csv', rows[:25])]:
        with (OUTPUT / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    save_json(OUTPUT / 'rankings.json', {
        'season': SEASON, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'games_used': number_of_games, 'teams_ranked': len(rows), 'rankings': rows
    })
    summary = ['# CBB Hybrid Rankings — 2026–27', '',
               'Experimental model: 50% opponent-adjusted scoring margin, 50% wins above expected.',
               f'Completed Division I matchups: {number_of_games}. Teams with games: {len(rows)}.', '',
               '| Rank | Team | Record | Hybrid | Predictive | Résumé |',
               '|---:|---|---:|---:|---:|---:|']
    for r in rows[:25]:
        summary.append(f"| {r['rank']} | {r['team']} | {r['wins']}-{r['losses']} | {r['hybrid']:.2f} | {r['predictive']:.2f} | {r['resume']:.2f} |")
    (OUTPUT / 'TOP25.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')


def main():
    games, teams = update_data()
    names, finals = standardize(games, teams)
    rows = calculate(names, finals)
    write_outputs(rows, len(finals))
    print(f'Success: {len(rows)} teams, {len(finals)} completed D-I games')
    for r in rows[:25]:
        print(f"{r['rank']:>2}. {r['team']:<25} {r['wins']}-{r['losses']}  {r['hybrid']:.2f}")


if __name__ == '__main__':
    main()
