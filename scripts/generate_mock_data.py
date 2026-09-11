"""
Generate a small synthetic dataset that matches the SCHEMA of the real
Kaggle "Customer Support on Twitter" file (twcs.csv):

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

This is NOT real data and NOT a substitute for it. It exists so that:
  1. every downstream module (loader, cleaner, thread reconstruction,
     intent discovery, baselines, evaluation, tests) can be built and
     verified to actually run, in an environment with no internet access.
  2. the code path is proven correct before you point it at the real CSV.

Brand names here are deliberately fictional (MockBrandA / MockBrandB) so
there is no confusion with real company data. Run this once; it writes
data/mock_raw/twcs_mock.csv.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.config import MOCK_RAW_DIR

random.seed(42)

BRANDS = {
    "MockBrandA": {
        "handle": "@MockBrandASupport",
        "issue_templates": [
            # login_problem
            ("login", "I can't log into my {brand} account, it keeps saying invalid password even after reset."),
            ("login", "Locked out of my {brand} account for 2 days now, no reset email arriving."),
            ("login", "{brand} login button spins endlessly and never loads my profile."),
            ("login", "Didn't receive the 2FA SMS verification code to log into my {brand} account."),
            ("login", "Why does the {brand} app keep logging me out every 5 minutes?"),
            # billing_issue
            ("billing", "{brand} charged me twice for the same order #{n}, need this fixed."),
            ("billing", "Why was I billed {amount} when my plan is supposed to be {amount2}?"),
            ("billing", "Saw an unexpected extra fee of {amount} on my latest {brand} invoice."),
            ("billing", "I got charged a monthly renewal fee of {amount} even though I paused my subscription."),
            ("billing", "{brand} billed my credit card after I switched payment methods."),
            # refund_request
            ("refund", "Requested a refund on order #{n} a week ago, still nothing."),
            ("refund", "Can I get a refund for {brand} order #{n}, item arrived damaged."),
            ("refund", "Returned my package 10 days ago for order #{n}, when will my refund be credited?"),
            ("refund", "Need a full refund for order #{n} because the item sent was completely wrong."),
            ("refund", "Support promised my refund for #{n} in 3 days, it has been two weeks now."),
            # delivery_problem
            ("delivery", "My {brand} order #{n} says delivered but I never received it."),
            ("delivery", "Package from {brand} has been 'in transit' for 9 days, tracking not updating."),
            ("delivery", "Tracking says delivery attempted for order #{n}, but I was home all day."),
            ("delivery", "Where is my order #{n}? Estimated delivery date was 4 days ago."),
            ("delivery", "Courier delivered my package for order #{n} to the wrong street address."),
            # technical_problem
            ("technical", "The {brand} app crashes every time I open the payments tab."),
            ("technical", "{brand} website checkout is throwing a 500 error on step 3."),
            ("technical", "Cannot apply discount codes at checkout on the {brand} app, getting an error."),
            ("technical", "Your iOS app freezes completely on the home screen after latest update."),
            ("technical", "Search bar on {brand} website is broken and returns 0 results for everything."),
            # cancellation
            ("cancellation", "How do I cancel my {brand} subscription? Can't find the option anywhere."),
            ("cancellation", "Please cancel my order #{n} immediately before it ships out."),
            ("cancellation", "Want to cancel my auto-renew membership with {brand}."),
            ("cancellation", "Cancel button is disabled in my account settings, please assist."),
            # complaint
            ("complaint", "Extremely disappointed with {brand} support response time, this is the 3rd time I've written in."),
            ("complaint", "Worst customer service experience ever from {brand}, rude agents and no resolution."),
            ("complaint", "I have been transferred between 4 different agents and none of you are resolving my case."),
            ("complaint", "Waiting on hold for 2 hours for order #{n}, absolutely unacceptable support."),
            # realistic ambiguous / multi-intent / boundary cases
            ("billing", "{brand} charged me for a renewal after I already cancelled, and now the app won't even load."),
            ("refund", "Order #{n} never arrived and support keeps closing my ticket without a refund, 3rd time asking."),
            ("technical", "Can't log in to check on my order #{n}, app just spins forever."),
            ("cancellation", "Your product arrived broken for order #{n}, cancel my recurring contract immediately."),
            ("complaint", "Fourth time reporting this bug where order #{n} keeps failing, fix your platform!"),
            # high risk / safety / escalation cases
            ("login", "Someone hacked into my {brand} account and changed my password, please help me lock it down!"),
            ("billing", "I noticed fraudulent unauthorized charges from {brand} on my card, my account was compromised!"),
            ("complaint", "I demand to speak to a real human supervisor right now, your automated bots are useless!"),
            ("billing", "If {brand} does not resolve this unauthorized charge of {amount}, I will pursue legal action with my lawyer."),
        ],
        "resolution_templates": {
            "login": "Sorry for the trouble! Please try resetting your password from a private/incognito window - if it still fails DM us your registered email so we can look into your account.",
            "billing": "We're sorry about that. Please DM us your order number and the last 4 digits of the card used so we can investigate the duplicate charge.",
            "delivery": "Apologies for the delay! Please DM your order number and zip code so we can trace the package with the carrier.",
            "refund": "So sorry for the wait. Please DM your order number and we'll escalate the refund to our billing team right away.",
            "technical": "Thanks for flagging this - please try updating to the latest app version, and DM us your device model if the crash continues.",
            "cancellation": "You can cancel anytime from Account > Subscription > Cancel. DM us if you don't see that option and we'll help directly.",
            "complaint": "We're really sorry to hear that. Please DM your case/order number so a senior agent can personally follow up.",
        },
    },
    "MockBrandB": {
        "handle": "@MockBrandBCare",
        "issue_templates": [
            ("login", "{brand} keeps logging me out every few minutes, super annoying."),
            ("billing", "{brand} auto-renewed my subscription without any warning email, want a refund."),
            ("delivery", "Still waiting on my {brand} order #{n}, it's been 12 days."),
            ("technical", "{brand} app won't load past the splash screen on iOS 17."),
            ("feature_question", "Does {brand} support exporting my data to CSV?"),
            ("complaint", "Third time contacting {brand} about the same bug, no fix yet."),
        ],
        "resolution_templates": {
            "login": "Sorry about that! Please clear your app cache and reinstall - if the issue continues, DM your account email.",
            "billing": "Apologies for the surprise renewal. DM your order ID and we'll process a refund review.",
            "delivery": "Sorry for the delay - DM your order number so we can chase this with our logistics partner.",
            "technical": "We're aware of an iOS 17 issue - please try reinstalling the latest build. DM your device model if it persists.",
            "feature_question": "Yes! You can export data from Settings > Data > Export CSV.",
            "complaint": "We're sorry this keeps happening. DM your case number so we can escalate internally.",
        },
    },
}


def build_rows() -> list[dict]:
    rows = []
    tweet_id = 1
    for brand, cfg in BRANDS.items():
        n_conversations = 1350 if brand == "MockBrandA" else 450
        for i in range(n_conversations):
            intent, template = random.choice(cfg["issue_templates"])
            text = template.format(
                brand=brand, n=random.randint(10000, 99999),
                amount=f"${random.randint(10,80)}.99", amount2=f"${random.randint(5,50)}.99",
            )
            cust_id = f"cust_{brand}_{i}"
            in_id = tweet_id
            rows.append({
                "tweet_id": tweet_id, "author_id": cust_id, "inbound": True,
                "created_at": f"2019-0{random.randint(1,9)}-{random.randint(10,28):02d} 12:00:00",
                "text": text, "response_tweet_id": tweet_id + 1,
                "in_response_to_tweet_id": None,
            })
            tweet_id += 1
            # ~85% of mock conversations get an actual support response
            if random.random() < 0.85:
                reply = cfg["resolution_templates"].get(intent, "Thanks for reaching out, we'll look into this.")
                rows.append({
                    "tweet_id": tweet_id, "author_id": cfg["handle"], "inbound": False,
                    "created_at": f"2019-0{random.randint(1,9)}-{random.randint(10,28):02d} 12:30:00",
                    "text": reply, "response_tweet_id": None,
                    "in_response_to_tweet_id": in_id,
                })
                tweet_id += 1
    random.shuffle(rows)
    return rows


def main() -> None:
    rows = build_rows()
    df = pd.DataFrame(rows)
    out_path = Path(MOCK_RAW_DIR) / "twcs_mock.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} mock rows to {out_path}")
    print(df["author_id"].value_counts().head(10))


if __name__ == "__main__":
    main()
