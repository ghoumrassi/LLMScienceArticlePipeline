import logging
import os
from abc import ABCMeta, abstractclassmethod

import dateutil.parser
from bs4 import BeautifulSoup
from utils import getTextOrNull, database_connection, init_logging

init_logging()


def createParsedContent():
    # Connect to the database
    connection = database_connection()

    sel_cursor = connection.cursor(buffered=True)
    ins_cursor = connection.cursor(buffered=True)

    # Prepare the INSERT and UPDATE statements
    insert_stmt = "INSERT INTO clean_documents (document_id, document_title, document_subtitle, published_date, document) VALUES (%s, %s, %s, %s, %s)"
    update_stmt = "UPDATE raw_documents SET parsed = %s WHERE id = %s"

    # Run the SQL query
    # TODO proper filtering for Science Daily
    sel_cursor.execute("SELECT document_id, sitemap_link, document_html FROM raw_documents")
    row = sel_cursor.fetchone()
    # Iterate over the query results
    while row is not None:
        article_id, sitemap_link, content = row
        try:
            if sitemap_link.startswith('https://sciencedaily.com/'):
                result = parseScienceDaily(content)
            elif sitemap_link.startswith('https://scitechdaily.com/'):
                pass
            elif sitemap_link.startswith('https://arstechnica.com/'):
                pass
            elif sitemap_link.startswith('https://phys.org/'):
                pass
            else:
                continue
            ins_cursor.execute(
                insert_stmt, 
                (
                    article_id, 
                    result['headline'], 
                    result['subtitle'], 
                    result['published_date'], 
                    result['text']
                )
            )
            ins_cursor.execute(update_stmt, (1, article_id))
        except:
            ins_cursor.execute(update_stmt, (-1, article_id))
            pass
        connection.commit()
        row = sel_cursor.fetchone()

    sel_cursor.close()
    ins_cursor.close()
    connection.close()



def parseScienceDaily(article_html):
    # TODO augment database and parsing function to extract more features
    # - topics
    # - abstract
    # etc.
    soup = BeautifulSoup(article_html, features='lxml')

    dd = {}
    
    text_container = soup.find(id='text')
    if not text_container:

        #TODO bad article
        pass

    # Get paragraphs
    paragraphs = text_container.find_all('p')
    text = ''
    total_chars = 0
    sections = [0]
    for i, p in enumerate(paragraphs):
        if p.find('strong') and i:
            sections.append(total_chars)
        ptext = getTextOrNull(p)
        if ptext:
            total_chars += len(ptext)
            text += ptext + '\n'
    
    dd['headline'] = getTextOrNull(soup.find(id='headline'))
    dd['subtitle'] = getTextOrNull(soup.find(id='subtitle'))
    dd['published_date'] = dateutil.parser.parse(
        getTextOrNull(soup.find(id='date_posted')))
    # dd['summary'] = getTextOrNull(soup.find(id='abstract'))
    # dd['topics'] = getTextOrNull(soup.find(id='related_terms'))
    dd['text'] = text
    dd['sections'] = sections

    return dd


if __name__ == '__main__':
    # dd = parseScienceDaily(r"C:\Users\aghou\Downloads\sd_test.html")
    # print(dd)
    createParsedContent()