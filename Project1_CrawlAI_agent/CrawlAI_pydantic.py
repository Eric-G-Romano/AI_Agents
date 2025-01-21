import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase Client and OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY'))

@dataclass
class ProcessedChunk:
    url: str
    chunk_number int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chuck_text(text:str, chuck_size:int=5000) -> List[str]:
    """Split text into chucks of chuck_size, accounting for code blocks and paragraphs.

    Args:
        text (str): The input text(string) to be divided into chunks.
        chuck_size (int, optional): The maximum size of each chunk. Defaults to 5000.

    Returns:
        List[str]: A list of strings, where each string is a chunk of the original text.
            """
            chunks = []
            current_chunk = []
            start = 0
            text_length = len(text)
            
            while start < text_length:
                end = start + chuck_size
                # Find the next newline character after the end of the chunk. Handle Edge case: Last chunk
                if end >= text_length:
                    chunks.append(text[start:].strip())
                    break
                # Extract a chuck and check for Boundaries.
                chunk =  text[start:end]
                code_block = chunk.rfind('```')
                # if a code block is found and its located past 30% of the chunk size, 
                # adjust the end position to align with the boundary of the code block.
                if code_block != -1 and code_block > 0.3*chuck_size:
                    end = start + code_block
                    
                # Adjust for Paragraph Boundaries
                elif '\n\n' in chunk:
                    last_break = chunk.rfind('\n\n')
                    if last_break > 0.3*chuck_size: # Only break if paragraph is past 30% of the chunk size
                        end = start + last_break
                        
                # Adjust for Sentence Boundaries
                elif '.' in chunk:
                    last_period = chunk.rfind('.')
                    if last_period > 0.3*chuck_size: # Only break if sentence is past 30% of the chunk size
                        end = start + last_period + 1
                
                # Clean Up and Store the Chunk
                chuck = text[start:end].strip()
                if chuck:
                    chunks.append(chuck)
                
                # Update the start position for the next chunk
                start = max(start + 1, end)
            
            return chunks
        
async def get_title_and_summary(chunk: str, url: str) -> Dict[str,str]:
    """Get the title and summary of a chunk of text using OpenAI's GPT-3 API.

    Args:
        chunk (str): The input text(string) to be summarized.
        url (str): The URL of the page from which the text was extracted.

    Returns:
        Dict[str,str]: A dictionary containing the title and summary of the chunk.
    """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative.""""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-40-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Limit the content to 1000 characters
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message['content'])
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error proccessing title", "summary": "Error processing summary"}
        
async def get_embedding(text: str) -> List[float]:
    """Get the embedding of a text using OpenAI's GPT-3 API.

    Args:
        text (str): The input text(string) to be embedded.

    Returns:
        List[float]: A list of floats representing the embedding of the text.
    """
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embeddings
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536 # Return zero vector on error
        
async def process_chunk(chunk: str, url: str, chunk_number: int) -> ProcessedChunk:
    """Process a chunk of text by extracting the title, summary, content, metadata and embedding.

    Args:
        chunk (str): The input text(string) to be processed.
        url (str): The URL of the page from which the text was extracted.
        chunk_number (int): The index of the chunk in the original text.

    Returns:
        ProcessedChunk: A dataclass containing the processed information of the chunk.
    """
   # Get the title and summary of the chunk
    title_summary = await get_title_and_summary(chunk, url)
    
   # Get the embedding of the chunk
    embedding = await get_embedding(chunk) 
   
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=title_summary["title"],
        summary=title_summary["summary"],
        content=chunk, # Store the original chunk
        metadata=metadata,
        embedding=embedding)

async def insert_chunks(chunk: ProcessedChunk):
    """Insert a processed chunk into the Supabase database.

    Args:
        chunk (ProcessedChunk): The processed chunk to be inserted.
    """
    try:
        response = await supabase.table("chunks").insert([
            {
                "url": chunk.url,
                "chunk_number": chunk.chunk_number,
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }
        ])
        if response["status"] == 201:
            print(f"Chunk {chunk.chunk_number} inserted successfully.")
        else:
            print(f"Error inserting chunk {chunk.chunk_number}: {response}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel.

    Args:
        url (str): The URL of the document.
        markdown (str): The markdown content of the document.
    """
    #split the markdown into chunks
    chunks = chuck_text(markdown)
    
    # Process and store each chunk in parallel
    task = [process_chunk(chunk, url, i) for i, chunk in enumerate(chunks)]
    
    processed_chunks = await asyncio.gather(*task)
    
    # store chunks in parallel
    insert_task = [insert_chunks(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_task)
    
async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawler_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        # Create a semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawler_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get the list of URLs for the Pydantic AI documentation."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML sitemap
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs for the Pydantic AI documentation
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found.")
        return
    
    print(f"Found {len(urls)} URLs.")
    await crawl_parallel(urls)
    
if __name__ == "__main__":
    asyncio.run(main())
    
