import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. owner/dataset")
    parser.add_argument("--out", default=os.path.join("data", "kaggle"), help="Output directory")
    args = parser.parse_args()

    config_dir = os.path.join(os.getcwd(), ".kaggle")
    os.environ["KAGGLE_CONFIG_DIR"] = config_dir
    os.makedirs(config_dir, exist_ok=True)

    token_path = os.path.join(config_dir, "kaggle.json")
    if not os.path.exists(token_path):
        print("Missing Kaggle API token.")
        print("Create .kaggle/kaggle.json in this project directory with your Kaggle username/key.")
        print("Expected path:")
        print(f"  {token_path}")
        return 2

    os.makedirs(args.out, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=args.out, unzip=True, quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
