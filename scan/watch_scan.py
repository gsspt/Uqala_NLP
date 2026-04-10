import json, pathlib, time
prog = pathlib.Path(__file__).parent / 'scan_progress.json'
print("Surveillance du scan (Ctrl+C pour arrêter)...")
while True:
    if prog.exists():
        try:
            s = json.load(open(prog, encoding='utf-8'))
            print(f"  {len(s['done_files'])} fichiers | {s['total_akhbars']:,} akhbars | {len(s['candidates'])} candidats", end='\r', flush=True)
        except Exception:
            pass
    time.sleep(10)
