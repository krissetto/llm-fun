'''for scraping stuff'''

import asyncio
import base64
import time
import xml.etree.ElementTree as ET

from datetime import datetime
from typing import Dict, Tuple, List

import aiohttp

from bs4 import (
    BeautifulSoup as BS,
    SoupStrainer
)

import db


async def get_docs_to_embed() -> dict[str, str]:
    '''
        Get docs from the docker documentation website.
        
        Returns a {url, url_contents} dict
    '''
    urls = await get_urls_from_sitemap("https://docs.docker.com")

    # save to db for future checks
    res = await db.save_sitemap(urls)

    if len(res) == 0:
        print("There is nothing new to scrape")
        return {}

    scraping_task_results = await scrape_urls([url for url, _ in res.items()])
    url_content_dict = parse_contents_from_urls(scraping_task_results)
    return url_content_dict


async def get_urls_from_sitemap(base_site_url: str) -> Dict[str,datetime]:
    '''
        Gets the sitemap for the docs website.
        Returns a list of (url, last_modified) tuples
    '''
    print(f"Getting list of all urls on {base_site_url} based on sitemap xml")
    async with aiohttp.ClientSession() as client:
        async with client.get(f"{base_site_url}/sitemap.xml") as response:
            sitemap_xml = await response.text()
    root = ET.fromstring(sitemap_xml)

    return {
        c[0].text: datetime.strptime(" ".join(c[1].text.split(" ")[:2]), "%Y-%m-%d")
        for c in root
        if c[0].text is not None
        and c[1].text is not None
    }


async def scrape_urls(urls: List[str]) -> List[Tuple[str, bytes]]:
    '''a'''
    print(f"grabbing contents from {len(urls)} url")
    async with aiohttp.ClientSession() as client:
        scraping_coros = [_get_page_content(client, url) for url in urls] 
        start_time = time.perf_counter()
        task_results = await asyncio.gather(*scraping_coros)
    print(f"grabbing contents from {len(urls)} urls took {time.perf_counter() - start_time:0.4f} seconds")

    return task_results


async def _get_page_content(client: aiohttp.ClientSession, url: str) -> Tuple[str, bytes]:
    '''
        Gets the main text from the url. 
        Returns a tuple containing the url and the content
    '''
    async with client.get(url) as response:
        content = await response.content.read()
    return (url, content)


def parse_contents_from_urls(scraping_task_results: List[Tuple[str, bytes]]) -> Dict[str, str]:
    '''
        Uses beautifulsoup to parse the first <article> tag in the html contents for each url.
        Returns a dict with the url as a key and the text contents of <article></article> as the value.
    '''
    print(f"parsing contents snatched from {len(scraping_task_results)} urls...")

    result: dict[str, str] = dict()

    start_time = time.perf_counter()

    for url, html_text in scraping_task_results:
        article_text, ok = _parse_response(html_text)
        if not ok:
            continue
        result[url] = article_text

    print(f"parsing {len(scraping_task_results)} url contents took {time.perf_counter() - start_time:0.4f} seconds")

    return result


def _parse_response(page_text: bytes) -> tuple[str, bool]:
    article_tags = SoupStrainer("article")
    soup = BS(page_text, 'lxml', parse_only=article_tags)

    # Replace all h1 and h2 blocks with their string contents, followed by a newlines
    h1_nodes = soup.find_all("h1")
    for node in h1_nodes:
        # Get the content of the <h1> tag
        h1_content = node.get_text()
        # Replace the node with the content
        node.replace_with(f"{h1_content}\n\n")

    h2_nodes = soup.find_all("h2")
    for node in h2_nodes:
        # Get the content of the <h2> tag
        h2_content = node.get_text()
        # Replace the node with the content
        node.replace_with(f"{h2_content}\n\n")

    # Remove the node with class="syntax-light"
    # these should contain code blocks
    code_block_nodes = soup.select('.syntax-light')

    for node in code_block_nodes:
        # right above code_block_nodes there should be a button.
        # get the x-data attribute, which is a json object, and extract the 'code' key
        # this contains the actual code of the code block, encoded in base64
        btn_node = node.find_previous_sibling('button')
        btn_node_data = btn_node['x-data']

        # fucking shit parsing. the json in the attribute is invalid
        encoded_code = btn_node_data.split("code: '")[1].split("'")[0]
        decoded_code = base64.b64decode(encoded_code).decode('utf-8')
        # replace the node containing both the btn and
        # the code block node with the actual decoded code
        node.find_parent().find_parent().replace_with(f"\n\n```\n{decoded_code}\n```\n\n")

    # Remove the "table of contents"
    toc_node = soup.find(text="Table of contents")
    if toc_node is not None:
        # go up two levels and remove from there
        toc_node.find_parent().find_parent().decompose()

    # Replace all <code></code> blocks with their string contents
    inline_code_block_nodes = soup.find_all("code")
    for node in inline_code_block_nodes:
        # Get the content of the <code> tag and use it to replace the node
        node.replace_with(f"`{node.get_text()}`")

    # Replace all <a></a> blocks with md links
    a_nodes = soup.find_all("a")
    for node in a_nodes:
        # Get the contents and link of the <a> tag and use them to replace the node
        a_text = node.get_text()
        a_link = node.get("href")
        node.replace_with(f"[{a_text}]({a_link})")

    article = soup.find("article")

    return (article.get_text(strip=False, separator=" "), True) if article is not None else ('', False)
