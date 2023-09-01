from Scweet.scweet import scrape


data = scrape(words=["chinese core socialist values"], since="2021-01-01", until="2023-07-01", from_account = None,
              interval=5, headless=False, display_type="Top", save_images=False, lang="en", limit=100,
              resume=True , filter_replies=False, proximity=False, minlikes=0, cookie_file='cookie.json',
              time_delay=90, search_type='and')