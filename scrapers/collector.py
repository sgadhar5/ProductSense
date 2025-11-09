from scrapers.twitter_scraper import scrape_twitter
from scrapers.instagram_scraper import scrape_instagram
from scrapers.playstore_scraper import scrape_playstore

def scrape_all():
    all_data = []
    all_data += scrape_twitter(limit=40)
    all_data += scrape_instagram(limit=30)
    all_data += scrape_playstore(limit=40)
    print(f"✅ Collected {len(all_data)} posts total.")
    return all_data
