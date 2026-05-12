import argparse
import os

from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("data", "hf", "cuad"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    local_path = hf_hub_download(
        repo_id="theatticusproject/cuad",
        repo_type="dataset",
        filename=os.path.join("CUAD_v1", "master_clauses.csv"),
        local_dir=args.out,
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

