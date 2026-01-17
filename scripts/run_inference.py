import sys
import json
from pathlib import Path

# -----------------------------
# Ensure project root is in path
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.ollama_client import OllamaClient  # noqa: E402


def load_prompt(version: str):
    base = ROOT / "prompts" / version
    system = (base / "system.txt").read_text()
    user = (base / "user.txt").read_text()
    return system, user


def load_dataset():
    path = ROOT / "data" / "golden" / "dataset_v1.json"
    return json.loads(path.read_text())["samples"]


def main():
    config = json.loads((ROOT / "runs" / "run_config_v1.json").read_text())

    system_prompt, user_template = load_prompt(config["prompt_version"])

    client = OllamaClient(
        model=config["model"],
        temperature=config["temperature"],
        top_p=config["top_p"],
    )

    outputs = []

    for sample in load_dataset():
        user_prompt = user_template.replace("{{input}}", sample["input"])
        result = client.generate(system_prompt, user_prompt)

        outputs.append(
            {
                "id": sample["id"],
                "input": sample["input"],
                "output": result["output"],
            }
        )

    output_dir = ROOT / "runs" / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "baseline_outputs_v1.json"
    output_path.write_text(json.dumps(outputs, indent=2))

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
