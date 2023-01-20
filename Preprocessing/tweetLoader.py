"""This module contains methods to access twitter text via TWitter API."""


import pandas as pd
import tweepy
from tqdm.auto import tqdm

from config import API_KEY, API_KEY_SECRET, BEARER_TOKEN, TOKEN, TOKEN_SECRET


class TweetLoader:
    """Class for fetching tweets identified by tweetID via Twitter API."""

    def __init__(self):
        """Initialize tweet loader."""
        self.client = tweepy.Client(
            bearer_token=BEARER_TOKEN,
            consumer_key=API_KEY,
            consumer_secret=API_KEY_SECRET,
            access_token=TOKEN,
            access_token_secret=TOKEN_SECRET,
            wait_on_rate_limit=True,
        )
        self.TWEET_LIMIT = 100

    def fetch_single_tweet(self, tweetID: str) -> str:
        """Fetch single tweet identified by tweetID."""
        tweet = self.client.get_tweets(ids=[tweetID])

        if not tweet.errors:
            return tweet.data[0].text
        else:
            return tweet.errors[0]["title"]

    def fetch_list(self, ids_list: list) -> pd.DataFrame:
        """Fetch list of tweet ids."""
        tweets_lst = []

        # batches according to maximal twitter api limit
        for tweet_batch in tqdm(
            self._batch(ids_list, batch_size=self.TWEET_LIMIT), total=len(ids_list) / self.TWEET_LIMIT
        ):
            tweets = self.client.get_tweets(ids=tweet_batch)

            if tweets.data is not None:
                tweets_lst.extend(tweets.data)

        return self._tweets_to_pandas(tweets_lst)

    def _tweets_to_pandas(self, lst) -> pd.DataFrame:
        """Fast way to load data into dataframe."""
        row_list = []
        for row in lst:
            dict1 = {}
            dict1.update({"tweetID": row.id, "text": row.text})
            row_list.append(dict1)

        return pd.DataFrame(row_list, columns=["tweetID", "text"])

    def _batch(self, lst, batch_size):
        """Create batches of fixed size from list of arbitrary length."""
        lst_length = len(lst)
        for idx in range(0, lst_length, batch_size):
            yield lst[idx : min(idx + batch_size, lst_length)]


if __name__ == "__main__":
    tweets = pd.read_csv(
        "/home/tomas/Documents/MediaBiasGroup/raw_datasets/uploaded/23_twitter_hatespeech/NAACL_SRW_2016.csv",
        header=None,
    )
    tweets = tweets[0].to_list()
    tweets = [str(t) for t in tweets]

    tl = TweetLoader()

    res = tl.fetch_list(tweets[:1000])
