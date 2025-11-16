#!/usr/bin/env python3
"""
Reddit post and comment scraper (Diaper-related discussions)

Fetch posts and comments containing keywords from specific subreddits via Reddit API, export as JSON.
Focuses on diaper-related discussions for competitive analysis and user needs research.

Usage:
    python scripts/reddit_subreddit_scraper.py --limit 100
    python scripts/reddit_subreddit_scraper.py --subreddits BabyBumps Parenting diapers --limit 200
    python scripts/reddit_subreddit_scraper.py --keywords diaper baby nappy --limit 150
"""

import os
import sys
import time
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    import requests
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependencies. Please run: pip install -r scripts/requirements.txt")
    print(f"Detail error: {e}")
    sys.exit(1)


class RedditAPIClient:
    """Reddit API client, handles authentication and API requests"""
    
    def __init__(self):
        # 加载环境变量
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'painly-scraper/1.0 (https://painly.ai)')
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.local"
            )
        
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.base_url = 'https://oauth.reddit.com'
        self.auth_url = 'https://www.reddit.com/api/v1/access_token'
        
        # Rate limiting: 60 requests/minute = 1 second/request
        self.rate_limit_delay = 1.0
        self.last_request_time = 0
        
    def authenticate(self) -> None:
        """Authenticate using client credentials flow"""
        try:
            auth = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            
            response = requests.post(
                self.auth_url,
                headers={
                    'Authorization': f'Basic {auth}',
                    'User-Agent': self.user_agent,
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                data={'grant_type': 'client_credentials'},
                timeout=30
            )
            
            response.raise_for_status()
            auth_data = response.json()
            
            self.access_token = auth_data['access_token']
            # Expire 5 minutes early for safety
            self.token_expiry = time.time() + auth_data['expires_in'] - 300
            
            print(f"[Authentication Successful] Token will expire in {auth_data['expires_in']} seconds")
            
        except Exception as e:
            raise RuntimeError(f"Reddit authentication failed: {e}")
    
    def ensure_authenticated(self) -> None:
        """Ensure a valid access token exists"""
        if not self.access_token or time.time() >= self.token_expiry:
            self.authenticate()
    
    def _wait_for_rate_limit(self) -> None:
        """Respect rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def make_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make API request"""
        self.ensure_authenticated()
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent,
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"[Rate Limited] Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.make_request(endpoint, params)
            
            # Handle authentication expiry
            if response.status_code == 401:
                print("[Authentication Expired] Re-authenticating...")
                self.access_token = None
                self.ensure_authenticated()
                return self.make_request(endpoint, params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"[Request Failed] {endpoint}: {e}")
            return None
    
    def get_subreddits(
        self, 
        mode: str = 'popular', 
        limit: int = 100,
        query: Optional[str] = None,
        after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get subreddit list
        
        Args:
            mode: 'popular', 'new', or 'search'
            limit: Number per request (max 100)
            query: Search query (only for search mode)
            after: Pagination marker
        """
        limit = min(limit, 100)  # Reddit API limit
        
        if mode == 'search':
            if not query:
                raise ValueError("Search mode requires query parameter")
            endpoint = '/subreddits/search'
            params = {'q': query, 'limit': limit}
        elif mode == 'popular':
            endpoint = '/subreddits/popular'
            params = {'limit': limit}
        elif mode == 'new':
            endpoint = '/subreddits/new'
            params = {'limit': limit}
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        if after:
            params['after'] = after
        
        return self.make_request(endpoint, params)
    
    def get_subreddit_about(self, subreddit: str) -> Optional[Dict[str, Any]]:
        """Get subreddit detailed information"""
        endpoint = f'/r/{subreddit}/about'
        result = self.make_request(endpoint)
        if result and 'data' in result:
            return result['data']
        return None
    
    def get_subreddit_rules(self, subreddit: str) -> List[Dict[str, Any]]:
        """Get subreddit rules"""
        endpoint = f'/r/{subreddit}/about/rules'
        result = self.make_request(endpoint)
        if result and 'rules' in result:
            return result['rules']
        return []

    def get_posts(
        self,
        subreddit: str,
        sort: str = 'hot',
        limit: int = 100,
        after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get post list from subreddit

        Args:
            subreddit: subreddit name
            sort: Sort method ('hot', 'new', 'top', 'rising')
            limit: Number per request (max 100)
            after: Pagination marker
        """
        limit = min(limit, 100)
        endpoint = f'/r/{subreddit}/{sort}'
        params = {'limit': limit}

        if after:
            params['after'] = after

        return self.make_request(endpoint, params)

    def get_post_comments(self, subreddit: str, post_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all comments from a post

        Args:
            subreddit: subreddit name
            post_id: post ID

        Returns:
            Comment list or None
        """
        endpoint = f'/r/{subreddit}/comments/{post_id}'
        result = self.make_request(endpoint)

        if result and isinstance(result, list) and len(result) >= 2:
            # Reddit API returns [post_data, comments_data]
            return result[1]
        return None


class RedditPostScraper:
    """Reddit post and comment scraper (focused on diaper-related discussions)"""

    def __init__(
        self,
        output_file: str,
        subreddits: List[str],
        keywords: List[str],
        batch_size: int = 10,
        max_comments_per_post: int = 50
    ):
        self.client = RedditAPIClient()
        self.output_file = output_file
        self.subreddits = subreddits
        self.keywords = [kw.lower() for kw in keywords]  # Convert to lowercase for matching
        self.batch_size = batch_size
        self.max_comments_per_post = max_comments_per_post

        # Collected data
        self.collected_posts: List[Dict[str, Any]] = []
        self.total_posts_processed = 0
        self.total_posts_matched = 0
        self.total_comments_collected = 0
    
    def _contains_keywords(self, text: str) -> bool:
        """Check if text contains any keyword"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.keywords)

    def _extract_post_data(self, post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract post data from API response"""
        title = post_data.get('title', '')
        selftext = post_data.get('selftext', '')

        # Check if title or content contains keywords
        if not self._contains_keywords(title) and not self._contains_keywords(selftext):
            return None

        post_id = post_data.get('id', '')
        created_utc = post_data.get('created_utc', 0)
        created_date = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S') if created_utc else ''

        return {
            'id': post_id,
            'title': title,
            'content': selftext.strip(),
            'author': post_data.get('author', '[deleted]'),
            'subreddit': post_data.get('subreddit', ''),
            'score': post_data.get('score', 0),
            'num_comments': post_data.get('num_comments', 0),
            'url': f"https://reddit.com{post_data.get('permalink', '')}",
            'created_date': created_date,
            'created_utc': int(created_utc),
            'comments': []  # Will be populated later
        }

    def _extract_comments(self, comments_data: Dict[str, Any], max_count: int) -> List[Dict[str, Any]]:
        """Recursively extract comment data (including nested comments)"""
        comments = []

        if not comments_data or 'data' not in comments_data:
            return comments

        children = comments_data['data'].get('children', [])

        for child in children:
            if len(comments) >= max_count:
                break

            if child.get('kind') != 't1':  # t1 = comment
                continue

            comment_data = child.get('data', {})
            body = comment_data.get('body', '')

            # Only collect comments containing keywords
            if not self._contains_keywords(body):
                continue

            created_utc = comment_data.get('created_utc', 0)
            created_date = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S') if created_utc else ''

            comment = {
                'id': comment_data.get('id', ''),
                'author': comment_data.get('author', '[deleted]'),
                'body': body.strip(),
                'score': comment_data.get('score', 0),
                'created_date': created_date,
                'created_utc': int(created_utc),
            }
            comments.append(comment)

            # Recursively extract replies
            if 'replies' in comment_data and comment_data['replies']:
                replies = self._extract_comments(
                    comment_data['replies'],
                    max_count - len(comments)
                )
                comments.extend(replies)

        return comments

    def _write_batch(self) -> None:
        """Batch write collected data to JSON"""
        if not self.collected_posts:
            return

        # Read existing data (if file exists)
        existing_data = {"metadata": {}, "posts": []}
        if Path(self.output_file).exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"[Warning] Unable to read existing file: {e}")

        # Append new data
        existing_data['posts'].extend(self.collected_posts)

        # Update metadata
        existing_data['metadata'] = {
            'subreddits': self.subreddits,
            'keywords': self.keywords,
            'total_posts': len(existing_data['posts']),
            'total_comments': sum(len(p['comments']) for p in existing_data['posts']),
            'scraped_at': datetime.now().isoformat(),
        }

        # Write file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"[Write] Saved {len(self.collected_posts)} posts to JSON")
        self.collected_posts.clear()
    
    def scrape(self, posts_per_subreddit: int = 100, sort: str = 'hot') -> None:
        """
        Start scraping posts and comments

        Args:
            posts_per_subreddit: Number of posts to scrape per subreddit
            sort: Sort method ('hot', 'new', 'top', 'rising')
        """
        print(f"\n{'='*60}")
        print(f"Starting Reddit Post and Comment Scraping")
        print(f"Target Subreddits: {', '.join(self.subreddits)}")
        print(f"Keyword Filter: {', '.join(self.keywords)}")
        print(f"Scraping per subreddit: {posts_per_subreddit} posts")
        print(f"Sort method: {sort}")
        print(f"Max comments per post: {self.max_comments_per_post}")
        print(f"Batch write: Every {self.batch_size} posts")
        print(f"Output file: {self.output_file}")
        print(f"{'='*60}\n")

        try:
            for subreddit in self.subreddits:
                print(f"\n[Scraping] r/{subreddit}")
                after = None
                posts_fetched = 0

                pbar = tqdm(
                    total=posts_per_subreddit,
                    desc=f"r/{subreddit}",
                    unit="posts"
                )

                while posts_fetched < posts_per_subreddit:
                    # Get post list
                    result = self.client.get_posts(
                        subreddit=subreddit,
                        sort=sort,
                        limit=min(100, posts_per_subreddit - posts_fetched),
                        after=after
                    )

                    if not result or 'data' not in result:
                        print(f"\n[Complete] r/{subreddit} no more data")
                        break

                    data = result['data']
                    children = data.get('children', [])

                    if not children:
                        print(f"\n[Complete] r/{subreddit} no more posts")
                        break

                    # Process each post
                    for child in children:
                        if posts_fetched >= posts_per_subreddit:
                            break

                        post_data = child.get('data', {})
                        self.total_posts_processed += 1
                        posts_fetched += 1

                        # Extract post data (includes keyword filtering)
                        extracted_post = self._extract_post_data(post_data)

                        if extracted_post:
                            # Get comments
                            post_id = extracted_post['id']
                            comments_result = self.client.get_post_comments(
                                subreddit=subreddit,
                                post_id=post_id
                            )

                            if comments_result:
                                comments = self._extract_comments(
                                    comments_result,
                                    self.max_comments_per_post
                                )
                                extracted_post['comments'] = comments
                                self.total_comments_collected += len(comments)

                            self.collected_posts.append(extracted_post)
                            self.total_posts_matched += 1

                            # Write when batch size is reached
                            if len(self.collected_posts) >= self.batch_size:
                                self._write_batch()

                        pbar.update(1)
                        pbar.set_postfix({
                            'Matched': self.total_posts_matched,
                            'Comments': self.total_comments_collected
                        })

                    # Get next page marker
                    after = data.get('after')
                    if not after:
                        print(f"\n[Complete] r/{subreddit} reached last page")
                        break

                pbar.close()

            # Write remaining data
            self._write_batch()

        except KeyboardInterrupt:
            print("\n\n[Interrupted] User cancelled operation, saving collected data...")
            self._write_batch()
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()
            self._write_batch()
            raise

        print(f"\n{'='*60}")
        print(f"Scraping Complete!")
        print(f"Total posts processed: {self.total_posts_processed}")
        print(f"Posts matching keywords: {self.total_posts_matched}")
        print(f"Comments collected: {self.total_comments_collected}")
        print(f"Match rate: {self.total_posts_matched/self.total_posts_processed*100:.1f}%")
        print(f"Output file: {self.output_file}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape diaper-related posts and comments from Reddit API and export as JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Scrape 100 posts with default settings
  python scripts/reddit_subreddit_scraper.py --limit 100

  # Specify specific subreddits
  python scripts/reddit_subreddit_scraper.py --subreddits BabyBumps Parenting diapers --limit 200

  # Custom keywords
  python scripts/reddit_subreddit_scraper.py --keywords diaper nappy pampers --limit 150

  # Custom output file
  python scripts/reddit_subreddit_scraper.py --output evaluation/my_diaper_data.json --limit 100
        """
    )

    parser.add_argument(
        '--subreddits',
        type=str,
        nargs='+',
        default=['BabyBumps', 'Parenting', 'diapers', 'beyondthebump', 'NewParents'],
        help='List of subreddits to scrape (default: BabyBumps Parenting diapers beyondthebump NewParents)'
    )

    parser.add_argument(
        '--keywords',
        type=str,
        nargs='+',
        default=['diaper', 'diapers', 'baby', 'nappy', 'coterie', 'pampers', 'huggies', 'disposable'],
        help='Keyword filter list (default: diaper diapers baby nappy coterie pampers huggies disposable)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of posts to scrape per subreddit (default: 100)'
    )

    parser.add_argument(
        '--sort',
        type=str,
        choices=['hot', 'new', 'top', 'rising'],
        default='hot',
        help='Post sort method (default: hot)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: evaluation/reddit_diaper_posts_TIMESTAMP.json)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch write size (default: 10)'
    )

    parser.add_argument(
        '--max-comments-per-post',
        type=int,
        default=50,
        help='Max comments to collect per post (default: 50)'
    )

    args = parser.parse_args()

    # Generate default output filename
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Ensure evaluation folder exists
        eval_dir = Path(__file__).parent.parent / 'evaluation'
        eval_dir.mkdir(exist_ok=True)
        args.output = str(eval_dir / f'reddit_diaper_posts_{timestamp}.json')

    try:
        scraper = RedditPostScraper(
            output_file=args.output,
            subreddits=args.subreddits,
            keywords=args.keywords,
            batch_size=args.batch_size,
            max_comments_per_post=args.max_comments_per_post
        )

        scraper.scrape(
            posts_per_subreddit=args.limit,
            sort=args.sort
        )

    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

