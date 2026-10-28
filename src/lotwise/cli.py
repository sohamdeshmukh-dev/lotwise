import argparse
import json
from pathlib import Path

from lotwise.models import Account
from lotwise.optimizer import Rebalancer
from lotwise.synthetic import generate_accounts


def main():
    parser = argparse.ArgumentParser(description="Generate and review simulated SMA proposals")
    parser.add_argument(
        "--input", type=Path, help="Account JSON; defaults to a seeded synthetic account"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    account = (
        Account.model_validate_json(args.input.read_text())
        if args.input
        else generate_accounts(1)[0]
    )
    proposal = Rebalancer().propose(account)
    output = json.dumps(proposal.model_dump(), indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
