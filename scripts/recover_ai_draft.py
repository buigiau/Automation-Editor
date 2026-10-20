"""Export a complete draft from a failed AI run's saved review, without inference."""
import argparse

from autoedit.ai.pipeline import recover_draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', help='Path to the failed job ai-review.json')
    parser.add_argument('--output', required=True, help='Destination for the recovered bundle')
    args = parser.parse_args()
    result = recover_draft(args.review, args.output)
    print(result['plan_path'])


if __name__ == '__main__':
    main()
