import os
import json
from datetime import datetime, timedelta
import pytz
import logging
import mysql.connector
import feedparser
from watchtower import CloudWatchLogHandler
from dateutil.parser import parse

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s: %(message)s'
)

# logger = logging.getLogger()
# logger.addHandler(CloudWatchLogHandler())

# Define the timezone to use
tz = pytz.timezone('UTC')

# Define the MySQL database configuration
db_config = {
    'host': 'newtechtomorrow.ce9xaem0xjah.us-east-1.rds.amazonaws.com',
    'port': 3306,
    'user': 'admin',
    'password': 'Keras8102-1996',
    'database': 'autoscience'
}


def lambda_handler(event, context):
    return {
        'statusCode': 200,
        'body': process_feeds()
    }

def get_published_date(entry):
    """
    Extracts the published date from the feedparser entry object and returns it
    as a datetime object with timezone information.

    Args:
        entry: A feedparser entry object.

    Returns:
        A datetime object representing the published date with timezone
        information.
    """
    if 'published_parsed' in entry:
        return datetime(*entry['published_parsed'][:6])
    else:
        datetime_object = parse(entry.published)
    return datetime_object

def is_published_in_last_24_hours(published_date):
    """
    Determines whether the given published date falls within the last 24 hours.

    Args:
        published_date: A datetime object representing the published date with
        timezone information.

    Returns:
        True if the published date falls within the last 24 hours, False
        otherwise.
    """
    one_day_ago = datetime.now() - timedelta(days=1)
    return published_date > one_day_ago

def extract_data_from_entry(entry):
    """
    Extracts the headline, URL, and published date from the feedparser entry
    object, if the entry was published in the last 24 hours.

    Args:
        entry: A feedparser entry object.

    Returns:
        A tuple containing the headline, URL, and published date, or None if the
        entry was not published in the last 24 hours.
    """
    published_date = get_published_date(entry)
    if is_published_in_last_24_hours(published_date):
        headline = entry.title
        url = entry.link
        return (headline, url, published_date)
    return None

def insert_data_into_database(conn, cursor, headline, url, published_date):
    """
    Inserts the given headline, URL, and published date into the MySQL database,
    if a row with the same URL does not already exist.

    Args:
        conn: A MySQL database connection object.
        cursor: A MySQL database cursor object.
        headline: A string representing the headline of the article.
        url: A string representing the URL of the article.
        published_date: A datetime object representing the published date with
        timezone information.
    """
    query = 'SELECT url FROM rss_links WHERE url=%s'
    cursor.execute(query, (url,))
    result = cursor.fetchone()
    if result is None:
        insert_query = '''
        INSERT INTO rss_links (headline, url, published_date) VALUES (%s, %s, %s)
        '''
        cursor.execute(insert_query, (headline, url, published_date))
        conn.commit()
    else:
        logging.warning(f'URL already exists in database: {url}')

def process_entry(conn, cursor, entry):
    """
    Processes a single feedparser entry object, extracting the relevant data and
    inserting it into the MySQL database.

    Args:
        conn: A MySQL database connection object.
        cursor: A MySQL database cursor object.
        entry: A feedparser entry object.
    """
    try:
        data = extract_data_from_entry(entry)
        if data is not None:
            headline, url, published_date = data
            insert_data_into_database(
                conn, cursor, headline, url, published_date)
    except Exception as e:
        logging.error(
            f'Error extracting data from entry: {entry.link}, {str(e)}')

def process_feed(conn, cursor, rss_url):
    """
    Processes a single RSS feed, extracting the relevant data and inserting it
    into the MySQL database.

    Args:
        conn: A MySQL database connection object.
        cursor: A MySQL database cursor object.
        rss_url: A string representing the URL of the RSS feed.
    """
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        process_entry(conn, cursor, entry)


def process_feeds():
    """
    Iterates over a list of RSS feed URLs and extracts the relevant data from
    each entry before inserting it into a MySQL database.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
    except mysql.connector.Error as error:
        logging.error(f'MySQL error: {error}')
        
    try:
        with open('rssfeeds.txt', 'r') as f:
            rss_urls = f.read().splitlines()
        for rss_url in rss_urls:
            process_feed(conn, cursor, rss_url)
    except mysql.connector.Error as error:
        logging.error(f'Error: {error}')
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
    
    return "Successfully extracted RSS information."


if __name__ == '__main__':
    process_feeds()