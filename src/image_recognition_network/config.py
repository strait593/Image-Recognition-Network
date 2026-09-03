from pathlib import Path

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing a 'data' folder.")
