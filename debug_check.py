# Copier dans C:\Users\diaya\les cours\ensiasd\stage\projet\fleet_api\
# puis : python debug_check.py

import sqlite3, os

project = r"C:\Users\diaya\les cours\ensiasd\stage\projet"

print("=" * 55)
print("  RECHERCHE DE TOUTES LES BD SQLite")
print("=" * 55)

for root, dirs, files in os.walk(project):
    # Ignorer __pycache__ et .git
    dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', 'node_modules')]
    for f in files:
        if f.endswith(".db"):
            full = os.path.join(root, f)
            size = os.path.getsize(full)
            print(f"\n📁 {full}  ({size} octets)")
            try:
                conn = sqlite3.connect(full)
                tables = [t[0] for t in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()]
                print(f"   Tables : {tables}")
                if "gps_locations" in tables:
                    rows = conn.execute(
                        "SELECT immatriculation, latitude, longitude, "
                        "chauffeur_login, updated_at FROM gps_locations"
                    ).fetchall()
                    print(f"   ✅ gps_locations → {len(rows)} position(s)")
                    for r in rows:
                        print(f"      🚑 {r[0]} | lat={r[1]:.5f} lng={r[2]:.5f} | {r[3]} | {r[4]}")
                conn.close()
            except Exception as e:
                print(f"   ❌ Erreur : {e}")