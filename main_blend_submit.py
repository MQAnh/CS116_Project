import argparse
import pickle
from collections import Counter
from pathlib import Path


def load_submission(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_submission(submission, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(submission, f)
    print(f"Saved blended submission to: {path}")


def top_frequent_items(submission, top_n):
    if top_n <= 0:
        return set()
    counts = Counter()
    for items in submission.values():
        counts.update(items)
    return {item for item, count in counts.most_common(top_n)}


def blend_submissions(
    base_submission,
    extra_submission,
    k=10,
    base_keep=9,
    extra_limit=1,
    blocked_extra_items=None,
):
    blocked_extra_items = set() if blocked_extra_items is None else set(blocked_extra_items)
    blended = {}

    for customer_id, base_items in base_submission.items():
        extra_items = extra_submission.get(customer_id, [])
        items = []

        for item_id in base_items[:base_keep]:
            if item_id not in items and len(items) < k:
                items.append(item_id)

        n_extra = 0
        for item_id in extra_items:
            if len(items) >= k or n_extra >= extra_limit:
                break
            if item_id in blocked_extra_items or item_id in items:
                continue
            items.append(item_id)
            n_extra += 1

        for item_id in base_items:
            if len(items) >= k:
                break
            if item_id not in items:
                items.append(item_id)

        for item_id in extra_items:
            if len(items) >= k:
                break
            if item_id not in items:
                items.append(item_id)

        blended[customer_id] = items[:k]

    return blended


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="save_submit/submission_8.35_points.pkl")
    parser.add_argument("--extra", default="save_submit/submission_8.13_points.pkl")
    parser.add_argument("--output", default="outputs/submission_blend.pkl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--base-keep", type=int, default=9)
    parser.add_argument("--extra-limit", type=int, default=1)
    parser.add_argument("--block-top-extra", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading base submission: {args.base}")
    base_submission = load_submission(args.base)
    print(f"Loading extra submission: {args.extra}")
    extra_submission = load_submission(args.extra)

    blocked_extra_items = top_frequent_items(extra_submission, args.block_top_extra)
    if blocked_extra_items:
        print(
            "Blocked most frequent extra items: "
            + ", ".join(sorted(blocked_extra_items)[:10])
            + (" ..." if len(blocked_extra_items) > 10 else "")
        )

    blended = blend_submissions(
        base_submission,
        extra_submission,
        k=args.k,
        base_keep=args.base_keep,
        extra_limit=args.extra_limit,
        blocked_extra_items=blocked_extra_items,
    )
    save_submission(blended, args.output)


if __name__ == "__main__":
    main()
