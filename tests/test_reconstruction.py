import pandas as pd

from src.data.threads import reconstruct_pairs


def _mini_df():
    return pd.DataFrame([
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "t",
         "text": "help me", "response_tweet_id": 2, "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "@Brand", "inbound": False, "created_at": "t",
         "text": "sure, DM us", "response_tweet_id": None, "in_response_to_tweet_id": 1},
        # a support tweet replying to another support tweet - must NOT become a pair
        {"tweet_id": 3, "author_id": "@Brand", "inbound": False, "created_at": "t",
         "text": "following up", "response_tweet_id": None, "in_response_to_tweet_id": 2},
        # an unanswered customer tweet
        {"tweet_id": 4, "author_id": "cust2", "inbound": True, "created_at": "t",
         "text": "anyone there", "response_tweet_id": None, "in_response_to_tweet_id": None},
    ])


def test_reconstruct_pairs_only_customer_to_brand():
    df = _mini_df()
    pairs, stats = reconstruct_pairs(df, "@Brand")
    assert len(pairs) == 1
    assert pairs.iloc[0]["customer_message"] == "help me"
    assert pairs.iloc[0]["support_response"] == "sure, DM us"
    assert stats.total_inbound == 2
    assert stats.usable_pairs == 1
    assert stats.unanswered_inbound == 1
