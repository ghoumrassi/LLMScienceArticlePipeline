import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from tqdm import tqdm

import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import datetime
import dateutil.parser

import utils


class SitemapCrawler:
    def __init__(self, sitemap_url, refresh=1/24, stale_check=7):
        self.sitemap_url = sitemap_url
        self.refresh = refresh
        self.stale_check = stale_check
        
        self.links = set()
        self.crawled_links = set()
        self.errored_links = set()

        self.sitemap_id = None

        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "\
            "AppleWebKit/537.36 (KHTML, like Gecko) "\
            "Chrome/109.0.0.0 Safari/537.36"
        self.headers = {
            "User-Agent": ua
        }

        self.cnx = utils.database_connection()
        self.cursor = self.cnx.cursor(buffered=True)

    def _load_sitemap_links(self):
        # check if sitemap exists in db and get sitemap_id
        get_sm_query = """
            SELECT sitemaps_id, last_refreshed, sitemap_stale
            FROM sitemaps 
            WHERE sitemap_url = %s
        """
        self.cursor.execute(
            get_sm_query, (self.sitemap_url,)
        )
        # If no rows returned, sitemap has not been accessed before
        sm_exists = bool(self.cursor.rowcount)

        
        if sm_exists:
            # Run the query results to get sitemap id and last refreshed date.
            self.sitemap_id, last_refresh, stale = self.cursor.fetchone()

            # Get the links from sitemap_links
            get_links_query = """
                SELECT sitemap_link FROM sitemap_links
                WHERE sitemap_id = %s
            """
            self.cursor.execute(
                get_links_query, 
                (self.sitemap_id,)
            )
            exist_links = set(self.cursor.fetchall())

            # Check if sitemap should be refreshed
            # This is the case if last refresh was longer ago than 
            # dictated in self.refresh argument.
            refresh_cond = last_refresh - datetime.datetime.now() >\
                datetime.timedelta(days=self.refresh)  
        else:
            exist_links = set()
            refresh_cond = None

        # We should re-run sitemap collection if sitemap isn't in db
        # or if it was refreshed too long ago.
        # Refresh sitemap links
        if not sm_exists or refresh_cond:
            # download sitemap
            response = requests.get(self.sitemap_url, headers=self.headers)
            refreshed_date = datetime.datetime.utcnow()\
                .replace(tzinfo=datetime.timezone.utc)
            soup = BeautifulSoup(response.text, 'xml')

            # Oldest possible date in datetime (UTC)
            sm_lastmod = datetime.datetime.min\
                .replace(tzinfo=datetime.timezone.utc)
            # extract links from sitemap
            new_links_w_date = []
            for url in soup.find_all("url"):
                loc = url.loc.text
                if url.lastmod:
                    date = dateutil.parser.parse(url.lastmod.text)
                else: 
                    # TODO think about how to set this
                    # Currently if no lastmod, we will never treat as stale.
                    date = refreshed_date
                new_links_w_date.append((loc, date))
                if date > sm_lastmod:
                    sm_lastmod = date
            new_links_w_date = set(new_links_w_date)

            staledays = datetime.timedelta(days=self.stale_check)
            stale = (refreshed_date - sm_lastmod) > staledays

            # Update sitemap row in db
            self.cursor.execute(
                """
                INSERT INTO sitemaps 
                (sitemap_url, site_host_url, last_refreshed, sitemap_stale) 
                VALUES (%(sm_link)s, %(sm_host)s, %(refresh_date)s, %(stale)s)
                ON DUPLICATE KEY UPDATE
                last_refreshed=%(refresh_date)s,
                sitemap_stale=%(stale)s
                """, 
                {
                    "sm_link": self.sitemap_url, 
                    "sm_host": urlparse(self.sitemap_url).netloc,
                    "refresh_date": refreshed_date,
                    "stale": stale,
                }
            )
            self.sitemap_id = self.cursor.lastrowid
        else:
            new_links_w_date = set()
        
        ## Get all new links + lastmod
        diff_links = set(
            filter(
                lambda x: x[0] not in exist_links, new_links_w_date))
        
        # If we have new links, they need adding to database
        if diff_links:
            # Add sitemap links to sitemap_links
            insert_links_query = """
                INSERT INTO sitemap_links
                (sitemap_link, sitemap_id, status, published_date)
                VALUES (%s, %s, %s, %s)
            """
            insert_vals = [
                (
                    link,
                    self.sitemap_id,
                    "awaiting",
                    lastmod
                )\
                for link, lastmod in diff_links
            ]
            self.cursor.executemany(insert_links_query, insert_vals)

        self.cnx.commit()


    def _load_links(self):
        # check if crawled links file already exists and if so, read links from it
        links_query = """
            SELECT sitemap_link FROM sitemap_links 
            WHERE sitemap_id = %s
            AND status = "awaiting"
        """
        self.cursor.execute(links_query, (self.sitemap_id,))
        self.links = set(self.cursor.fetchall())

    def _crawl(self):
        # Query to insert row in raw_documents table
        raw_docs_query = """
            INSERT INTO raw_documents
            (sitemap_id, sitemap_link, added_date, document_html)
            VALUES (%s, %s, %s, %s)
        """

        # Query to update row status in sitemap_links
        sml_update_query = """
            UPDATE sitemap_links
            SET 
            status = %s,
            document_id = %s
            WHERE sitemap_link = %s
        """

        # iterate over links to crawl
        for i, (link,) in enumerate(tqdm(self.links)):
            try:
                # send GET request to link
                response = requests.get(link, timeout=5, headers=self.headers)
                # check if link returns a successful status code
                if response.status_code == 200: 
                    # Insert doc in raw_documents table
                    self.cursor.execute(
                        raw_docs_query,
                        (
                            self.sitemap_id,
                            link,
                            datetime.datetime.now(),
                            response.content
                        )
                    )        
                    # Update status in sitemap_links table
                    self.cursor.execute(
                        sml_update_query, (
                            "crawled",
                            self.cursor.lastrowid,
                            link
                        )
                    )
                    
                else:
                    self.cursor.execute(
                        sml_update_query, (
                            "errored", 
                            None,
                            link
                        )
                    )
            except requests.exceptions.RequestException as e:
                self.cursor.execute(
                    sml_update_query, (                            
                        "errored", 
                        None,
                        link
                    )
                )
                print("Request Error:", e)
            except requests.exceptions.HTTPError as e:
                self.cursor.execute(
                    sml_update_query, ("errored", link)
                )
                print("HTTP Error:", e)
            time.sleep(1)

            # Commit every 10 links
            if (i+1) % 10 == 0:
                self.cnx.commit()
        self.cnx.commit()

    def main(self):
        self._load_sitemap_links()
        self._load_links()
        self._crawl()


if __name__ == "__main__":
    sitemaps = [
        "https://www.sciencedaily.com/sitemap-releases-2023.xml",
        "https://scitechdaily.com/post-sitemap31.xml",
        "https://arstechnica.com/sitemap-pt-post-2023-01.xml",
        "https://futurism.com/sitemaps/post-sitemap30.xml",
        "https://phys.org/sitemap/",
        "https://www.sciencenews.org/post-sitemap1.xml",
        "https://www.universetoday.com/post-sitemap26.xml",
        "https://www.livescience.com/sitemap-2023-01.xml",
        "https://www.nature.com/nphys/sitemap/2023/1/articles.xml",
        "https://physicsworld.com/post-sitemap17.xml"
    ]

    for sm in sitemaps:
        crawler = SitemapCrawler(
            sm
        )
        crawler.main()