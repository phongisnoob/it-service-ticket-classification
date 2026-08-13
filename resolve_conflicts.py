import os

for root, _, files in os.walk("."):
    if ".git" in root or ".venv" in root or "__pycache__" in root:
        continue
    for file in files:
        if not file.endswith(".py"):
            continue
        path = os.path.join(root, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if "<<<<<<<" not in content:
                continue

            lines = content.split("\n")
            new_lines = []
            i = 0
            while i < len(lines):
                if lines[i].startswith("<<<<<<< HEAD"):
                    ours_block = []
                    i += 1
                    while not lines[i].startswith("======="):
                        ours_block.append(lines[i])
                        i += 1
                    i += 1
                    theirs_block = []
                    while not lines[i].startswith(">>>>>>> origin/fix/task-8-mypy"):
                        theirs_block.append(lines[i])
                        i += 1

                    ours = "\n".join(ours_block)
                    theirs = "\n".join(theirs_block)

                    if "backend=" in ours and '"auto"' in ours:
                        # inference.py get_predictor args
                        new_lines.append('    backend: typing.Any = "auto",')
                        new_lines.append(") -> typing.Any:")
                    elif (
                        "ROOT_DIR = Path" in theirs
                        or "METRICS_DIR = ROOT_DIR" in theirs
                        or "FIGURE_DIR.mkdir" in theirs
                        or "MODEL_PATH =" in theirs
                        or "DATA_PATH =" in theirs
                        or "ARTIFACT_DIR =" in theirs
                        or "THRESHOLD_PATH =" in theirs
                        or "MODEL_SELECTION_PATH =" in theirs
                    ):
                        if "from src.paths import" in ours or "from src.hashing import" in ours:
                            new_lines.extend(ours_block)
                        elif "def main():" in ours and "def main() -> None:" in theirs:
                            new_lines.append("def main() -> None:")
                        elif "def get_selected_backend() -> str:" in theirs:
                            new_lines.extend(theirs_block)
                        else:
                            new_lines.extend(ours_block)
                    elif "def main():" in ours and "def main() -> None:" in theirs:
                        new_lines.append("def main() -> None:")
                    elif "def main" in ours and "# type: ignore" in theirs:
                        new_lines.extend(theirs_block)
                    elif "from typing import Any" in theirs and "ROOT_DIR" in ours:
                        new_lines.extend(ours_block)
                        new_lines.extend(theirs_block)
                    elif "def test_predict_returns_valid_structure" in ours:
                        new_lines.append(
                            "    def test_predict_returns_valid_structure(self, predictor: typing.Any) -> None:"
                        )
                    elif "def calculate_sha256" in ours:
                        new_lines.append("def calculate_sha256(path: typing.Any) -> typing.Any:")
                    elif "def forward" in ours:
                        new_lines.append("    def forward(self, x: typing.Any) -> typing.Any:")
                    elif "def __init__(self):" in ours and "def __init__(self) -> None:" in theirs:
                        new_lines.append("    def __init__(self) -> None:")
                    elif "# type: ignore" in theirs:
                        new_lines.extend(theirs_block)
                    elif "class TicketDataset(Dataset):" in ours:
                        new_lines.append("class TicketDataset(Dataset[typing.Any]):")
                    elif "__len__" in ours and "__getitem__" in ours:
                        new_lines.extend(theirs_block)
                    else:
                        if "def get_selected_backend() -> str:" in theirs:
                            new_lines.extend(theirs_block)
                        else:
                            new_lines.extend(theirs_block)
                else:
                    new_lines.append(lines[i])
                i += 1

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
        except Exception as e:
            print(f"Failed on {path}: {e}")
