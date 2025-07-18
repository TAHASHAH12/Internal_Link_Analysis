import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import networkx as nx
from urllib.parse import urljoin, urlparse
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import json
import re
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import warnings
import base64
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Configure Streamlit page
st.set_page_config(
    page_title="Universal PageRank SEO Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .category-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .priority-config {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .column-selector {
        background-color: #fff;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DynamicCategoryDetector:
    """Automatically detect and categorize pages based on URL patterns, content, and structure"""
    
    def __init__(self):
        self.business_keywords = {
            'product': ['product', 'products', 'shop', 'store', 'buy', 'purchase', 'item', 'catalog'],
            'service': ['service', 'services', 'solutions', 'consulting', 'support'],
            'content': ['blog', 'news', 'article', 'post', 'content', 'insights', 'resources'],
            'company': ['about', 'company', 'team', 'history', 'mission', 'vision'],
            'contact': ['contact', 'contactus', 'touch', 'location', 'address', 'phone'],
            'legal': ['privacy', 'terms', 'legal', 'policy', 'conditions', 'disclaimer'],
            'help': ['help', 'support', 'faq', 'guide', 'documentation', 'manual'],
            'user': ['login', 'register', 'signup', 'account', 'profile', 'dashboard'],
            'category': ['category', 'categories', 'tag', 'tags', 'topic', 'topics'],
            'search': ['search', 'results', 'find', 'query'],
            'finance': ['loans', 'mortgage', 'credit', 'banking', 'finance', 'investment', 'insurance'],
            'healthcare': ['health', 'medical', 'doctor', 'hospital', 'clinic', 'treatment'],
            'education': ['course', 'education', 'learn', 'training', 'tutorial', 'class'],
            'technology': ['software', 'app', 'tech', 'digital', 'cloud', 'api', 'development'],
            'ecommerce': ['cart', 'checkout', 'payment', 'order', 'shipping', 'delivery'],
            'real_estate': ['property', 'real-estate', 'house', 'apartment', 'rent', 'buy'],
            'automotive': ['car', 'auto', 'vehicle', 'parts', 'repair', 'dealer'],
            'travel': ['travel', 'hotel', 'flight', 'booking', 'destination', 'tour'],
            'food': ['restaurant', 'food', 'menu', 'recipe', 'cooking', 'dining'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'show', 'event']
        }
        
        self.url_patterns = {}
        self.discovered_categories = set()
        self.category_keywords = {}
        
    def analyze_url_structure(self, urls):
        """Analyze URL structure to identify patterns and categories"""
        url_segments = []
        
        for url in urls:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if path:
                segments = [seg for seg in path.split('/') if seg and not seg.isdigit()]
                url_segments.extend(segments)
        
        # Count segment frequency
        segment_counts = Counter(url_segments)
        common_segments = [seg for seg, count in segment_counts.items() if count >= 2]
        
        return common_segments
    
    def extract_content_keywords(self, page_data):
        """Extract keywords from page content for better categorization"""
        all_text = []
        
        for url, data in page_data.items():
            if data.get('title'):
                all_text.append(data['title'])
            if data.get('meta_description'):
                all_text.append(data['meta_description'])
            if data.get('h1'):
                all_text.append(data['h1'])
        
        # Use TF-IDF to find important terms
        if all_text:
            try:
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)
                tfidf_matrix = vectorizer.fit_transform(all_text)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_terms = [feature_names[i] for i in mean_scores.argsort()[-20:][::-1]]
                
                return top_terms
            except:
                return []
        
        return []
    
    def categorize_url(self, url, title="", meta_desc="", h1=""):
        """Dynamically categorize a URL based on patterns and content"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Combine all text for analysis
        full_text = f"{path} {title} {meta_desc} {h1}".lower()
        
        # Check against business keywords
        category_scores = {}
        
        for category, keywords in self.business_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                category_scores[category] = score
        
        # Check URL patterns
        path_segments = [seg for seg in path.split('/') if seg]
        
        # Special handling for homepage
        if path in ['/', ''] or len(path_segments) == 0:
            return 'homepage'
        
        # Check for common patterns
        first_segment = path_segments[0] if path_segments else ''
        
        # Dynamic category detection based on URL structure
        if first_segment:
            # Check if this segment appears frequently (indicating a category)
            if first_segment in self.url_patterns:
                self.url_patterns[first_segment] += 1
            else:
                self.url_patterns[first_segment] = 1
        
        # Return best matching category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return best_category
        
        # If no match, use first URL segment as category
        return first_segment if first_segment else 'other'
    
    def auto_detect_categories(self, page_data):
        """Automatically detect website categories based on crawled data"""
        urls = list(page_data.keys())
        
        # Analyze URL structure
        common_segments = self.analyze_url_structure(urls)
        
        # Extract content keywords
        content_keywords = self.extract_content_keywords(page_data)
        
        # Categorize all pages
        categorized_pages = {}
        category_counts = Counter()
        
        for url, data in page_data.items():
            category = self.categorize_url(
                url,
                data.get('title', ''),
                data.get('meta_description', ''),
                data.get('h1', '')
            )
            
            categorized_pages[url] = category
            category_counts[category] += 1
        
        # Store discovered categories
        self.discovered_categories = set(category_counts.keys())
        
        return categorized_pages, category_counts, common_segments, content_keywords
    
    def get_business_value_mapping(self, categories, priority_pages_df=None):
        """Determine business value for each category"""
        business_value_mapping = {}
        
        # High value categories (business/revenue focused)
        high_value_indicators = [
            'product', 'service', 'finance', 'ecommerce', 'homepage',
            'real_estate', 'automotive', 'healthcare', 'education'
        ]
        
        # Low value categories (non-commercial)
        low_value_indicators = [
            'content', 'legal', 'help', 'category', 'search', 'user',
            'contact', 'company', 'entertainment'
        ]
        
        # Check priority pages if available
        priority_categories = set()
        if priority_pages_df is not None and not priority_pages_df.empty:
            for _, row in priority_pages_df.iterrows():
                url = row.get('Hub URL', '')
                if url:
                    category = self.categorize_url(url)
                    priority_categories.add(category)
        
        for category in categories:
            if category in priority_categories:
                business_value_mapping[category] = 'High Business Value'
            elif any(indicator in category.lower() for indicator in high_value_indicators):
                business_value_mapping[category] = 'High Business Value'
            elif any(indicator in category.lower() for indicator in low_value_indicators):
                business_value_mapping[category] = 'Low Business Value'
            else:
                business_value_mapping[category] = 'Medium Business Value'
        
        return business_value_mapping

def handle_priority_pages_upload():
    """Handle priority pages CSV upload with column selection"""
    st.subheader("📁 Priority Pages Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload Priority Pages CSV",
        type=['csv'],
        help="Upload CSV file containing your priority pages and target keywords"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show file preview
            st.markdown("**File Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection interface
            st.markdown('<div class="priority-config">', unsafe_allow_html=True)
            st.markdown("### 🎯 Configure Priority Pages Columns")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                st.markdown("**Select URL Column:**")
                url_column = st.selectbox(
                    "Choose the column containing page URLs",
                    options=df.columns.tolist(),
                    help="Select the column that contains the full URLs of your priority pages"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                st.markdown("**Select Target Keywords Column:**")
                keyword_column = st.selectbox(
                    "Choose the column containing target keywords",
                    options=['None'] + df.columns.tolist(),
                    help="Select the column that contains target keywords for each page (optional)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="column-selector">', unsafe_allow_html=True)
                st.markdown("**Select Category Column (Optional):**")
                category_column = st.selectbox(
                    "Choose the column containing page categories",
                    options=['None'] + df.columns.tolist(),
                    help="Select the column that contains page categories (optional)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Process the dataframe based on selections
            if url_column:
                priority_df = df.copy()
                
                # Rename columns based on user selection
                priority_df = priority_df.rename(columns={url_column: 'Hub URL'})
                
                if keyword_column != 'None':
                    priority_df = priority_df.rename(columns={keyword_column: 'Target Keywords'})
                else:
                    priority_df['Target Keywords'] = 'No keywords specified'
                
                if category_column != 'None':
                    priority_df = priority_df.rename(columns={category_column: 'Category'})
                else:
                    priority_df['Category'] = 'Not specified'
                
                # Clean the dataframe
                priority_df = priority_df[priority_df['Hub URL'].notna()]
                priority_df = priority_df[priority_df['Hub URL'].str.contains('http', na=False)]
                
                # Show processed data preview
                st.markdown("**Processed Priority Pages Preview:**")
                display_columns = ['Hub URL', 'Target Keywords', 'Category']
                st.dataframe(priority_df[display_columns].head(10), use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Priority Pages", len(priority_df))
                with col2:
                    unique_categories = priority_df['Category'].nunique()
                    st.metric("Unique Categories", unique_categories)
                with col3:
                    pages_with_keywords = len(priority_df[priority_df['Target Keywords'] != 'No keywords specified'])
                    st.metric("Pages with Keywords", pages_with_keywords)
                
                # Validation warnings
                if len(priority_df) == 0:
                    st.error("❌ No valid URLs found in the selected column. Please check your data.")
                    return None
                
                # Show category distribution
                if category_column != 'None':
                    st.markdown("**Category Distribution:**")
                    category_counts = priority_df['Category'].value_counts()
                    for category, count in category_counts.head(10).items():
                        st.markdown(f"• **{category}**: {count} pages")
                
                return priority_df
            
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {str(e)}")
            st.info("💡 Please ensure your CSV file is properly formatted and contains valid URLs.")
            return None
    
    return None

class EnhancedWebCrawler:
    """Enhanced web crawler with better URL discovery and processing"""
    
    def __init__(self, max_pages=500, delay=1):
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls = set()
        self.internal_links = defaultdict(list)
        self.page_data = {}
        self.domain = None
        self.category_detector = DynamicCategoryDetector()
        
    def is_valid_url(self, url):
        """Check if URL is valid and belongs to the domain"""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc == self.domain and
                url not in self.visited_urls and
                not any(ext in url.lower() for ext in [
                    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', 
                    '.xml', '.ico', '.svg', '.webp', '.mp4', '.mp3', '.zip',
                    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.woff',
                    '.ttf', '.eot', '.otf', '.json', '.txt', '.log'
                ]) and
                not any(param in url.lower() for param in [
                    'mailto:', 'tel:', 'javascript:', 'ftp:', '#'
                ])
            )
        except:
            return False
    
    def extract_all_links(self, soup, base_url):
        """Extract all internal links from page with improved detection"""
        links = []
        
        # Extract from <a> tags
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            if self.is_valid_url(full_url):
                links.append(full_url)
        
        # Extract from navigation menus
        for nav in soup.find_all('nav'):
            for link in nav.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                if self.is_valid_url(full_url):
                    links.append(full_url)
        
        # Extract from sitemap references
        for link in soup.find_all('link', rel='sitemap'):
            href = link.get('href', '')
            if href:
                full_url = urljoin(base_url, href)
                if 'sitemap' in full_url.lower():
                    sitemap_links = self.extract_sitemap_urls(full_url)
                    links.extend(sitemap_links)
        
        return list(set(links))
    
    def extract_sitemap_urls(self, sitemap_url):
        """Extract URLs from sitemap"""
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                urls = []
                for loc in soup.find_all('loc'):
                    url = loc.text.strip()
                    if self.is_valid_url(url):
                        urls.append(url)
                return urls
        except:
            pass
        return []
    
    def extract_page_content(self, soup):
        """Extract comprehensive page content"""
        content = {
            'title': '',
            'meta_description': '',
            'h1': '',
            'h2_tags': [],
            'h3_tags': [],
            'word_count': 0,
            'links_count': 0,
            'images_count': 0,
            'canonical_url': '',
            'lang': 'en'
        }
        
        # Title
        title = soup.find('title')
        content['title'] = title.text.strip() if title else 'No Title'
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        content['meta_description'] = meta_desc.get('content', '') if meta_desc else 'No Description'
        
        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        content['canonical_url'] = canonical.get('href', '') if canonical else ''
        
        # Language
        html_tag = soup.find('html')
        content['lang'] = html_tag.get('lang', 'en') if html_tag else 'en'
        
        # H1 tag
        h1 = soup.find('h1')
        content['h1'] = h1.text.strip() if h1 else 'No H1'
        
        # H2 and H3 tags
        content['h2_tags'] = [h2.text.strip() for h2 in soup.find_all('h2')]
        content['h3_tags'] = [h3.text.strip() for h3 in soup.find_all('h3')]
        
        # Word count
        text = soup.get_text()
        content['word_count'] = len(text.split()) if text else 0
        
        # Links and images count
        content['links_count'] = len(soup.find_all('a', href=True))
        content['images_count'] = len(soup.find_all('img'))
        
        return content
    
    def crawl_page(self, url):
        """Crawl a single page and extract comprehensive data"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract page content
            content = self.extract_page_content(soup)
            
            # Extract internal links
            links = self.extract_all_links(soup, url)
            
            return {
                'url': url,
                'title': content['title'],
                'meta_description': content['meta_description'],
                'h1': content['h1'],
                'h2_tags': content['h2_tags'],
                'h3_tags': content['h3_tags'],
                'word_count': content['word_count'],
                'links_count': content['links_count'],
                'images_count': content['images_count'],
                'canonical_url': content['canonical_url'],
                'lang': content['lang'],
                'status': response.status_code,
                'links': links
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': 'Error',
                'meta_description': str(e),
                'h1': 'Error',
                'h2_tags': [],
                'h3_tags': [],
                'word_count': 0,
                'links_count': 0,
                'images_count': 0,
                'canonical_url': '',
                'lang': 'en',
                'status': 0,
                'links': []
            }
    
    def discover_initial_urls(self, start_url):
        """Discover initial URLs from homepage and sitemap"""
        urls = [start_url]
        
        try:
            # Try to find sitemap.xml
            sitemap_urls = [
                f"{start_url}/sitemap.xml",
                f"{start_url}/sitemap_index.xml",
                f"{start_url}/sitemap/",
                f"{start_url}/robots.txt"
            ]
            
            for sitemap_url in sitemap_urls:
                try:
                    response = requests.get(sitemap_url, timeout=10)
                    if response.status_code == 200:
                        if 'sitemap' in sitemap_url:
                            sitemap_links = self.extract_sitemap_urls(sitemap_url)
                            urls.extend(sitemap_links[:100])  # Limit sitemap URLs
                        elif 'robots.txt' in sitemap_url:
                            # Extract sitemap URLs from robots.txt
                            for line in response.text.split('\n'):
                                if line.lower().startswith('sitemap:'):
                                    sitemap_url = line.split(':', 1)[1].strip()
                                    sitemap_links = self.extract_sitemap_urls(sitemap_url)
                                    urls.extend(sitemap_links[:100])
                except:
                    continue
            
            return list(set(urls))
            
        except:
            return [start_url]
    
    def crawl_website(self, start_url):
        """Crawl entire website starting from URL with enhanced discovery"""
        self.domain = urlparse(start_url).netloc
        
        # Discover initial URLs
        initial_urls = self.discover_initial_urls(start_url)
        to_visit = initial_urls
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while to_visit and len(self.visited_urls) < self.max_pages:
            current_batch = to_visit[:5]
            to_visit = to_visit[5:]
            
            for url in current_batch:
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                    
                    page_data = self.crawl_page(url)
                    self.page_data[url] = page_data
                    
                    # Store internal links
                    self.internal_links[url] = page_data['links']
                    
                    # Add new URLs to visit
                    for link in page_data['links']:
                        if link not in self.visited_urls and link not in to_visit:
                            to_visit.append(link)
                    
                    # Update progress
                    progress = len(self.visited_urls) / min(self.max_pages, len(self.visited_urls) + len(to_visit))
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Crawled {len(self.visited_urls)} pages... Found {len(to_visit)} more URLs to visit")
                    
                    time.sleep(self.delay)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Crawling complete! Found {len(self.page_data)} pages with {sum(len(links) for links in self.internal_links.values())} internal links")
        
        return self.page_data, dict(self.internal_links)

class PageRankCalculator:
    """Calculate PageRank using NetworkX with enhanced algorithms"""
    
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def calculate_pagerank(self, internal_links):
        """Calculate PageRank using NetworkX with fallback methods"""
        G = nx.DiGraph()
        
        # Add all pages as nodes
        all_pages = set(internal_links.keys())
        for page in internal_links.keys():
            all_pages.update(internal_links[page])
        
        G.add_nodes_from(all_pages)
        
        # Add edges with weights
        for source, targets in internal_links.items():
            for target in targets:
                if target in all_pages:
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1)
        
        try:
            # Try standard PageRank
            pagerank_scores = nx.pagerank(
                G, 
                alpha=self.damping_factor,
                max_iter=self.max_iterations,
                tol=self.tolerance,
                weight='weight'
            )
        except:
            try:
                # Fallback to unweighted PageRank
                pagerank_scores = nx.pagerank(
                    G, 
                    alpha=self.damping_factor,
                    max_iter=self.max_iterations,
                    tol=self.tolerance
                )
            except:
                # Final fallback - equal distribution
                pagerank_scores = {node: 1.0/len(all_pages) for node in all_pages}
        
        return pagerank_scores

class UniversalSEOAnalyzer:
    """Universal SEO analyzer that works with any website"""
    
    def __init__(self, priority_pages_df=None):
        self.priority_pages_df = priority_pages_df
        self.category_detector = DynamicCategoryDetector()
        
    def analyze_pagerank_distribution(self, pagerank_scores, page_data):
        """Analyze PageRank distribution with dynamic categorization"""
        # Auto-detect categories
        categorized_pages, category_counts, common_segments, content_keywords = self.category_detector.auto_detect_categories(page_data)
        
        # Get business value mapping
        business_value_mapping = self.category_detector.get_business_value_mapping(
            categorized_pages.values(), 
            self.priority_pages_df
        )
        
        # Create analysis dataframe
        analysis_data = []
        
        for url, score in pagerank_scores.items():
            if url in page_data:
                data = page_data[url]
                category = categorized_pages.get(url, 'other')
                business_value = business_value_mapping.get(category, 'Medium Business Value')
                
                # Check if this is a priority page
                is_priority = False
                target_keywords = 'No keywords specified'
                priority_category = 'Not specified'
                
                if self.priority_pages_df is not None and not self.priority_pages_df.empty:
                    priority_match = self.priority_pages_df[self.priority_pages_df['Hub URL'] == url]
                    if not priority_match.empty:
                        is_priority = True
                        target_keywords = priority_match.iloc[0].get('Target Keywords', 'No keywords specified')
                        priority_category = priority_match.iloc[0].get('Category', 'Not specified')
                        business_value = 'High Business Value'  # Override for priority pages
                
                analysis_data.append({
                    'URL': url,
                    'PageRank': score,
                    'Category': category,
                    'Title': data.get('title', ''),
                    'H1': data.get('h1', ''),
                    'Word_Count': data.get('word_count', 0),
                    'Links_Count': data.get('links_count', 0),
                    'Images_Count': data.get('images_count', 0),
                    'Business_Value': business_value,
                    'Status': data.get('status', 0),
                    'Canonical_URL': data.get('canonical_url', ''),
                    'Language': data.get('lang', 'en'),
                    'Is_Priority': is_priority,
                    'Target_Keywords': target_keywords,
                    'Priority_Category': priority_category
                })
        
        df = pd.DataFrame(analysis_data)
        if len(df) > 0:
            df['PageRank_Normalized'] = df['PageRank'] * 1000
            df['Percentage'] = (df['PageRank'] / df['PageRank'].sum()) * 100
            df['Rank'] = df['PageRank'].rank(ascending=False)
            df = df.sort_values('PageRank', ascending=False)
        
        return df, category_counts, common_segments, content_keywords, business_value_mapping

class UniversalVisualizationEngine:
    """Create visualizations for any website with priority pages integration"""
    
    def create_category_discovery_chart(self, category_counts, common_segments):
        """Show discovered categories and URL patterns"""
        
        # Category distribution
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Discovered Categories', 'Common URL Patterns'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Categories pie chart
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        fig.add_trace(
            go.Pie(labels=categories, values=counts, name="Categories"),
            row=1, col=1
        )
        
        # URL patterns bar chart
        if common_segments:
            fig.add_trace(
                go.Bar(
                    x=common_segments[:10],
                    y=[category_counts.get(seg, 0) for seg in common_segments[:10]],
                    name="URL Patterns"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Website Structure Analysis",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_business_value_analysis(self, analysis_df):
        """Create business value analysis chart"""
        if len(analysis_df) == 0:
            return None
        
        business_summary = analysis_df.groupby('Business_Value').agg({
            'PageRank': 'sum',
            'Percentage': 'sum',
            'URL': 'count'
        }).reset_index()
        
        fig = px.bar(
            business_summary,
            x='Business_Value',
            y='Percentage',
            color='Business_Value',
            title='PageRank Distribution by Business Value',
            labels={'Percentage': 'PageRank Percentage (%)', 'Business_Value': 'Business Value'},
            color_discrete_map={
                'High Business Value': '#4ecdc4',
                'Medium Business Value': '#ffd93d',
                'Low Business Value': '#ff6b6b'
            }
        )
        
        # Add value labels on bars
        for i, row in business_summary.iterrows():
            fig.add_annotation(
                x=row['Business_Value'],
                y=row['Percentage'],
                text=f"{row['Percentage']:.1f}%<br>({row['URL']} pages)",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            )
        
        return fig
    
    def create_priority_pages_detailed_analysis(self, analysis_df):
        """Create detailed priority pages analysis"""
        if len(analysis_df) == 0:
            return None
        
        priority_pages = analysis_df[analysis_df['Is_Priority'] == True]
        
        if len(priority_pages) == 0:
            return None
        
        fig = px.scatter(
            priority_pages,
            x='Rank',
            y='PageRank',
            size='Percentage',
            color='Priority_Category',
            hover_data=['URL', 'Title', 'Target_Keywords'],
            title='Priority Pages Performance Analysis',
            labels={'Rank': 'PageRank Rank', 'PageRank': 'PageRank Value'}
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_priority_vs_regular_comparison(self, analysis_df):
        """Create comparison between priority and regular pages"""
        if len(analysis_df) == 0:
            return None
        
        analysis_df['Page_Type'] = analysis_df['Is_Priority'].apply(
            lambda x: 'Priority Page' if x else 'Regular Page'
        )
        
        fig = px.scatter(
            analysis_df.head(100),
            x='Rank',
            y='PageRank',
            color='Page_Type',
            size='Percentage',
            hover_data=['URL', 'Category', 'Title', 'Target_Keywords'],
            title='Priority Pages vs Regular Pages Analysis',
            labels={'Rank': 'PageRank Rank', 'PageRank': 'PageRank Value'},
            color_discrete_map={'Priority Page': '#e74c3c', 'Regular Page': '#95a5a6'}
        )
        
        return fig
    
    def create_category_performance_matrix(self, analysis_df):
        """Create category performance matrix"""
        if len(analysis_df) == 0:
            return None
        
        category_summary = analysis_df.groupby('Category').agg({
            'PageRank': 'sum',
            'Percentage': 'sum',
            'URL': 'count',
            'Word_Count': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            category_summary,
            x='URL',
            y='Percentage',
            size='Word_Count',
            color='PageRank',
            hover_data=['Category'],
            title='Category Performance Matrix',
            labels={
                'URL': 'Number of Pages',
                'Percentage': 'PageRank Percentage (%)',
                'Word_Count': 'Average Word Count'
            },
            color_continuous_scale='viridis'
        )
        
        return fig
    
    def create_network_visualization(self, analysis_df, internal_links):
        """Create network visualization with priority pages highlighted"""
        if len(analysis_df) == 0:
            return None
        
        # Create network graph for top pages
        G = nx.DiGraph()
        
        top_pages = analysis_df.head(30)
        for _, row in top_pages.iterrows():
            G.add_node(row['URL'],
                      pagerank=row['PageRank'],
                      category=row['Category'],
                      business_value=row['Business_Value'],
                      is_priority=row['Is_Priority'],
                      target_keywords=row['Target_Keywords'],
                      title=row['Title'][:50] + '...' if len(row['Title']) > 50 else row['Title'])
        
        # Add edges
        for source, targets in internal_links.items():
            if source in G.nodes:
                for target in targets:
                    if target in G.nodes:
                        G.add_edge(source, target)
        
        if len(G.nodes) == 0:
            return None
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_text.append(f"Category: {node_data['category']}<br>"
                           f"Title: {node_data['title']}<br>"
                           f"PageRank: {node_data['pagerank']:.4f}<br>"
                           f"Business Value: {node_data['business_value']}<br>"
                           f"Priority: {'Yes' if node_data['is_priority'] else 'No'}<br>"
                           f"Keywords: {node_data['target_keywords']}")
            
            # Color by priority status and business value
            if node_data['is_priority']:
                node_color.append('#e74c3c')  # Red for priority pages
            elif node_data['business_value'] == 'High Business Value':
                node_color.append('#4ecdc4')
            elif node_data['business_value'] == 'Low Business Value':
                node_color.append('#ff6b6b')
            else:
                node_color.append('#ffd93d')
            
            # Size by PageRank
            node_size.append(max(10, node_data['pagerank'] * 10000))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2)
            )
        )
        
        # Updated layout syntax
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text='Internal Link Network (Red = Priority Pages)',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="Node size = PageRank, Red = Priority Pages",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        return fig

def create_excel_download(analysis_df, category_counts, business_value_mapping, priority_pages_df=None, recommendations_df=None):
    """Create comprehensive Excel report with priority pages analysis"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main analysis
        analysis_df.to_excel(writer, sheet_name='PageRank_Analysis', index=False)
        
        # Priority pages specific analysis
        if priority_pages_df is not None and not priority_pages_df.empty:
            priority_analysis = analysis_df[analysis_df['Is_Priority'] == True]
            if not priority_analysis.empty:
                priority_analysis.to_excel(writer, sheet_name='Priority_Pages_Analysis', index=False)
            
            # Priority pages performance summary
            priority_summary = priority_analysis.groupby('Priority_Category').agg({
                'PageRank': ['sum', 'mean'],
                'Percentage': ['sum', 'mean'],
                'Rank': 'mean',
                'URL': 'count'
            }).round(3)
            priority_summary.to_excel(writer, sheet_name='Priority_Summary')
            
            # Keyword analysis
            keyword_analysis = priority_analysis[priority_analysis['Target_Keywords'] != 'No keywords specified']
            if not keyword_analysis.empty:
                keyword_analysis[['URL', 'Title', 'Target_Keywords', 'PageRank', 'Percentage', 'Rank']].to_excel(
                    writer, sheet_name='Keyword_Analysis', index=False
                )
        
        # Category summary
        category_summary = analysis_df.groupby('Category').agg({
            'PageRank': 'sum',
            'Percentage': 'sum',
            'URL': 'count',
            'Business_Value': 'first'
        }).sort_values('PageRank', ascending=False)
        category_summary.to_excel(writer, sheet_name='Category_Summary')
        
        # Business value summary
        business_summary = analysis_df.groupby('Business_Value').agg({
            'PageRank': 'sum',
            'Percentage': 'sum',
            'URL': 'count'
        }).round(3)
        business_summary.to_excel(writer, sheet_name='Business_Value_Summary')
        
        # Recommendations
        if recommendations_df is not None and not recommendations_df.empty:
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
        
        # Top pages by category
        top_pages_by_category = []
        for category in category_counts.keys():
            top_page = analysis_df[analysis_df['Category'] == category].head(1)
            if not top_page.empty:
                top_pages_by_category.append(top_page.iloc[0])
        
        if top_pages_by_category:
            top_pages_df = pd.DataFrame(top_pages_by_category)
            top_pages_df.to_excel(writer, sheet_name='Top_Pages_By_Category', index=False)
    
    return output.getvalue()

def generate_universal_recommendations(analysis_df, category_counts, business_value_mapping, priority_pages_df=None):
    """Generate recommendations with priority pages insights"""
    recommendations = []
    
    # Priority pages specific recommendations
    if priority_pages_df is not None and not priority_pages_df.empty:
        priority_pages = analysis_df[analysis_df['Is_Priority'] == True]
        
        # Low performing priority pages
        low_priority_pages = priority_pages[priority_pages['Percentage'] < 1]
        for _, row in low_priority_pages.iterrows():
            recommendations.append({
                'Issue': f"Priority page '{row['Title']}' receiving only {row['Percentage']:.1f}% of PageRank",
                'Page_URL': row['URL'],
                'Category': row['Category'],
                'Target_Keywords': row['Target_Keywords'],
                'Current_PageRank': f"{row['PageRank']:.4f}",
                'Current_Percentage': f"{row['Percentage']:.1f}%",
                'Current_Rank': f"#{int(row['Rank'])}",
                'Impact': 'Critical - Priority page underperforming',
                'Recommendation': f"Increase internal linking to this priority page targeting '{row['Target_Keywords']}'",
                'Priority': 'Immediate'
            })
    
    # Find low-value pages with high PageRank
    low_value_high_pr = analysis_df[
        (analysis_df['Business_Value'] == 'Low Business Value') & 
        (analysis_df['Percentage'] > 2)
    ].sort_values('PageRank', ascending=False)
    
    for _, row in low_value_high_pr.iterrows():
        recommendations.append({
            'Issue': f"Low-value {row['Category']} page receiving {row['Percentage']:.1f}% of PageRank",
            'Page_URL': row['URL'],
            'Category': row['Category'],
            'Target_Keywords': row['Target_Keywords'],
            'Current_PageRank': f"{row['PageRank']:.4f}",
            'Current_Percentage': f"{row['Percentage']:.1f}%",
            'Current_Rank': f"#{int(row['Rank'])}",
            'Impact': 'High - Non-revenue page getting ranking power',
            'Recommendation': f"Reduce internal links to {row['Category']} pages or implement nofollow",
            'Priority': 'Immediate' if row['Percentage'] > 5 else 'High'
        })
    
    # Find high-value pages with low PageRank
    high_value_low_pr = analysis_df[
        (analysis_df['Business_Value'] == 'High Business Value') & 
        (analysis_df['Percentage'] < 1) &
        (analysis_df['Is_Priority'] == False)  # Exclude priority pages as they're handled above
    ].sort_values('PageRank', ascending=True)
    
    for _, row in high_value_low_pr.iterrows():
        recommendations.append({
            'Issue': f"High-value {row['Category']} page only receiving {row['Percentage']:.1f}% of PageRank",
            'Page_URL': row['URL'],
            'Category': row['Category'],
            'Target_Keywords': row['Target_Keywords'],
            'Current_PageRank': f"{row['PageRank']:.4f}",
            'Current_Percentage': f"{row['Percentage']:.1f}%",
            'Current_Rank': f"#{int(row['Rank'])}",
            'Impact': 'High - Revenue page underoptimized',
            'Recommendation': f"Increase internal linking to {row['Category']} pages",
            'Priority': 'Immediate'
        })
    
    return pd.DataFrame(recommendations)

def display_priority_pages_insights(analysis_df, priority_pages_df):
    """Display detailed insights about priority pages"""
    if priority_pages_df is None or priority_pages_df.empty:
        return
    
    priority_pages = analysis_df[analysis_df['Is_Priority'] == True]
    
    if len(priority_pages) == 0:
        st.warning("⚠️ No priority pages found in the crawled data. Please check your URLs.")
        return
    
    st.markdown('<div class="priority-config">', unsafe_allow_html=True)
    st.markdown("### 🎯 Priority Pages Performance Analysis")
    
    # Priority pages metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Priority Pages Found", len(priority_pages))
    
    with col2:
        avg_rank = priority_pages['Rank'].mean()
        st.metric("Average Rank", f"#{int(avg_rank)}")
    
    with col3:
        total_priority_pr = priority_pages['Percentage'].sum()
        st.metric("Total Priority PageRank %", f"{total_priority_pr:.1f}%")
    
    with col4:
        avg_priority_pr = priority_pages['Percentage'].mean()
        st.metric("Average Priority PageRank %", f"{avg_priority_pr:.1f}%")
    
    # Priority pages performance table
    st.markdown("**Priority Pages Performance:**")
    priority_display = priority_pages[['URL', 'Title', 'Target_Keywords', 'PageRank', 'Percentage', 'Rank']].copy()
    priority_display['Rank'] = priority_display['Rank'].astype(int)
    priority_display = priority_display.sort_values('PageRank', ascending=False)
    st.dataframe(priority_display, use_container_width=True)
    
    # Priority pages by category
    if 'Priority_Category' in priority_pages.columns:
        st.markdown("**Priority Pages by Category:**")
        category_performance = priority_pages.groupby('Priority_Category').agg({
            'PageRank': 'sum',
            'Percentage': 'sum',
            'URL': 'count',
            'Rank': 'mean'
        }).round(2)
        category_performance['Avg_Rank'] = category_performance['Rank'].astype(int)
        st.dataframe(category_performance, use_container_width=True)
    
    # Keyword performance
    keywords_pages = priority_pages[priority_pages['Target_Keywords'] != 'No keywords specified']
    if not keywords_pages.empty:
        st.markdown("**Pages with Target Keywords:**")
        keyword_display = keywords_pages[['URL', 'Title', 'Target_Keywords', 'PageRank', 'Percentage']].copy()
        keyword_display = keyword_display.sort_values('PageRank', ascending=False)
        st.dataframe(keyword_display, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_category_insights(category_counts, business_value_mapping, common_segments, content_keywords):
    """Display insights about discovered categories"""
    
    st.markdown('<div class="category-info">', unsafe_allow_html=True)
    st.markdown("### 🔍 Website Structure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Discovered Categories:**")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            business_value = business_value_mapping.get(category, 'Medium Business Value')
            emoji = "🟢" if business_value == "High Business Value" else "🟡" if business_value == "Medium Business Value" else "🔴"
            st.markdown(f"{emoji} **{category}**: {count} pages ({business_value})")
    
    with col2:
        st.markdown("**🔗 Common URL Patterns:**")
        for segment in common_segments[:10]:
            st.markdown(f"• `/{segment}/`")
        
        if content_keywords:
            st.markdown("**📝 Key Content Terms:**")
            for keyword in content_keywords[:8]:
                st.markdown(f"• {keyword}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌐 Universal PageRank SEO Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced PageRank analysis with priority pages and target keywords configuration**")
    
    # Info box
    st.info("🚀 Upload your priority pages CSV, select the URL and keyword columns, then analyze any website's PageRank distribution!")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Crawling settings
    st.sidebar.subheader("Enhanced Crawling Settings")
    max_pages = st.sidebar.slider("Maximum Pages to Crawl", 50, 2000, 500)
    delay = st.sidebar.slider("Delay Between Requests (seconds)", 0.5, 3.0, 1.0)
    
    # PageRank settings
    st.sidebar.subheader("PageRank Settings")
    damping_factor = st.sidebar.slider("Damping Factor", 0.1, 0.95, 0.85)
    max_iterations = st.sidebar.slider("Maximum Iterations", 50, 200, 100)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌐 Website Analysis")
        website_url = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com",
            help="Enter the full URL of the website you want to analyze"
        )
        
        st.markdown("**Enhanced Features:**")
        st.markdown("• **Priority Pages Integration** - Upload CSV and map columns")
        st.markdown("• **Target Keywords Analysis** - Track keyword performance")
        st.markdown("• **Advanced Categorization** - AI-powered page classification")
        st.markdown("• **Comprehensive Reporting** - Excel export with priority insights")
    
    with col2:
        # Priority pages upload and configuration
        priority_pages_df = handle_priority_pages_upload()
    
    # Analysis button
    if st.button("🚀 Start Enhanced PageRank Analysis", type="primary", disabled=not website_url):
        if not website_url:
            st.error("Please enter a website URL")
            return
        
        try:
            parsed_url = urlparse(website_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                st.error("Please enter a valid URL with http:// or https://")
                return
        except:
            st.error("Invalid URL format")
            return
        
        # Initialize components
        crawler = EnhancedWebCrawler(max_pages, delay)
        pagerank_calc = PageRankCalculator(damping_factor, max_iterations)
        viz_engine = UniversalVisualizationEngine()
        
        # Step 1: Enhanced Crawling
        st.subheader("🕷️ Step 1: Enhanced Website Crawling & URL Discovery")
        st.info("🔍 Discovering URLs from homepage, sitemaps, navigation, and internal links...")
        
        with st.spinner("Crawling website with enhanced discovery..."):
            page_data, internal_links = crawler.crawl_website(website_url)
        
        if not page_data:
            st.error("Failed to crawl website. Please check the URL and try again.")
            return
        
        st.markdown(f'<div class="success-box">✅ Successfully discovered and crawled {len(page_data)} pages with {sum(len(links) for links in internal_links.values())} internal links</div>', unsafe_allow_html=True)
        
        # Step 2: PageRank Calculation
        st.subheader("📊 Step 2: Advanced PageRank Calculation")
        with st.spinner("Calculating PageRank with enhanced algorithms..."):
            pagerank_scores = pagerank_calc.calculate_pagerank(internal_links)
        
        st.success(f"✅ Calculated PageRank for {len(pagerank_scores)} pages")
        
        # Step 3: Universal SEO Analysis
        st.subheader("🔍 Step 3: AI-Powered SEO Analysis with Priority Pages")
        with st.spinner("Analyzing PageRank distribution with priority pages integration..."):
            seo_analyzer = UniversalSEOAnalyzer(priority_pages_df)
            analysis_df, category_counts, common_segments, content_keywords, business_value_mapping = seo_analyzer.analyze_pagerank_distribution(pagerank_scores, page_data)
        
        if len(analysis_df) == 0:
            st.error("No data to analyze")
            return
        
        # Display category insights
        display_category_insights(category_counts, business_value_mapping, common_segments, content_keywords)
        
        # Display priority pages insights
        display_priority_pages_insights(analysis_df, priority_pages_df)
        
        # Display key metrics
        st.subheader("📈 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pr = analysis_df['PageRank'].sum()
            st.metric("Total PageRank", f"{total_pr:.4f}")
        
        with col2:
            high_value_pr = analysis_df[analysis_df['Business_Value'] == 'High Business Value']['Percentage'].sum()
            st.metric("High Value Pages %", f"{high_value_pr:.1f}%")
        
        with col3:
            low_value_pr = analysis_df[analysis_df['Business_Value'] == 'Low Business Value']['Percentage'].sum()
            st.metric("Low Value Pages %", f"{low_value_pr:.1f}%")
        
        with col4:
            priority_pr = analysis_df[analysis_df['Is_Priority'] == True]['Percentage'].sum()
            st.metric("Priority Pages %", f"{priority_pr:.1f}%")
        
        # Visualizations
        st.subheader("📊 Comprehensive PageRank Analysis")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Website Structure", 
            "Business Value", 
            "Priority Pages", 
            "Priority vs Regular", 
            "Category Performance",
            "Network Graph"
        ])
        
        with tab1:
            fig_discovery = viz_engine.create_category_discovery_chart(category_counts, common_segments)
            if fig_discovery:
                st.plotly_chart(fig_discovery, use_container_width=True)
            
            st.subheader("Discovered Categories")
            category_df = pd.DataFrame([
                {
                    'Category': cat, 
                    'Page_Count': count, 
                    'Business_Value': business_value_mapping.get(cat, 'Medium'),
                    'Percentage': f"{(count/len(analysis_df)*100):.1f}%"
                }
                for cat, count in category_counts.items()
            ]).sort_values('Page_Count', ascending=False)
            st.dataframe(category_df, use_container_width=True)
        
        with tab2:
            fig_business = viz_engine.create_business_value_analysis(analysis_df)
            if fig_business:
                st.plotly_chart(fig_business, use_container_width=True)
            
            business_summary = analysis_df.groupby('Business_Value').agg({
                'PageRank': 'sum',
                'Percentage': 'sum',
                'URL': 'count'
            }).round(3)
            st.dataframe(business_summary, use_container_width=True)
        
        with tab3:
            fig_priority_detailed = viz_engine.create_priority_pages_detailed_analysis(analysis_df)
            if fig_priority_detailed:
                st.plotly_chart(fig_priority_detailed, use_container_width=True)
            else:
                st.info("No priority pages found in the crawled data")
        
        with tab4:
            fig_priority_comparison = viz_engine.create_priority_vs_regular_comparison(analysis_df)
            if fig_priority_comparison:
                st.plotly_chart(fig_priority_comparison, use_container_width=True)
        
        with tab5:
            fig_performance = viz_engine.create_category_performance_matrix(analysis_df)
            if fig_performance:
                st.plotly_chart(fig_performance, use_container_width=True)
            
            category_performance = analysis_df.groupby('Category').agg({
                'PageRank': 'sum',
                'Percentage': 'sum',
                'URL': 'count',
                'Word_Count': 'mean'
            }).sort_values('PageRank', ascending=False)
            st.dataframe(category_performance, use_container_width=True)
        
        with tab6:
            fig_network = viz_engine.create_network_visualization(analysis_df, internal_links)
            if fig_network:
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("Network visualization requires more interconnected pages")
        
        # Recommendations
        st.subheader("🎯 Priority-Focused SEO Recommendations")
        
        recommendations_df = generate_universal_recommendations(analysis_df, category_counts, business_value_mapping, priority_pages_df)
        
        if not recommendations_df.empty:
            # Group recommendations by priority
            immediate_actions = recommendations_df[recommendations_df['Priority'] == 'Immediate']
            high_priority = recommendations_df[recommendations_df['Priority'] == 'High']
            
            if not immediate_actions.empty:
                st.markdown("### 🚨 Immediate Action Required")
                st.dataframe(immediate_actions[['Issue', 'Target_Keywords', 'Current_Percentage', 'Current_Rank', 'Recommendation']], use_container_width=True)
            
            if not high_priority.empty:
                st.markdown("### ⚠️ High Priority Items")
                st.dataframe(high_priority[['Issue', 'Target_Keywords', 'Current_Percentage', 'Current_Rank', 'Recommendation']], use_container_width=True)
        else:
            st.success("🎉 No major PageRank issues found!")
        
        # Download section
        st.subheader("💾 Download Complete Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare Excel download
            excel_data = create_excel_download(analysis_df, category_counts, business_value_mapping, priority_pages_df, recommendations_df)
            
            st.download_button(
                label="📊 Download Complete Excel Report",
                data=excel_data,
                file_name=f"priority_pagerank_analysis_{parsed_url.netloc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV download
            csv_data = analysis_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Data",
                data=csv_data,
                file_name=f"pagerank_data_{parsed_url.netloc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Summary insights
        st.subheader("💡 Key Insights for Your Website")
        
        insights = []
        
        # Priority pages insights
        if priority_pages_df is not None and not priority_pages_df.empty:
            priority_pages = analysis_df[analysis_df['Is_Priority'] == True]
            if not priority_pages.empty:
                avg_priority_rank = priority_pages['Rank'].mean()
                insights.append(f"🎯 **Priority Pages**: {len(priority_pages)} priority pages found, average rank #{int(avg_priority_rank)}")
                
                low_performing_priority = priority_pages[priority_pages['Percentage'] < 1]
                if not low_performing_priority.empty:
                    insights.append(f"🔴 **Critical**: {len(low_performing_priority)} priority pages receiving less than 1% PageRank")
        
        # Website type detection
        if any(cat in ['product', 'ecommerce', 'shop'] for cat in category_counts.keys()):
            insights.append("🛒 **E-commerce Site Detected**: Focus on product pages and reduce blog/content PageRank")
        elif any(cat in ['content', 'blog', 'news'] for cat in category_counts.keys()):
            insights.append("📰 **Content Site Detected**: Balance content with conversion pages")
        elif any(cat in ['service', 'company'] for cat in category_counts.keys()):
            insights.append("🏢 **Corporate Site Detected**: Optimize service pages and reduce informational content")
        
        # Performance insights
        high_value_pr = analysis_df[analysis_df['Business_Value'] == 'High Business Value']['Percentage'].sum()
        low_value_pr = analysis_df[analysis_df['Business_Value'] == 'Low Business Value']['Percentage'].sum()
        
        if low_value_pr > 30:
            insights.append(f"🔴 **Critical Issue**: {low_value_pr:.1f}% of PageRank is going to low-value pages")
        
        if high_value_pr < 20:
            insights.append(f"🟡 **Optimization Opportunity**: Only {high_value_pr:.1f}% of PageRank reaches high-value pages")
        
        # Category insights
        top_category = max(category_counts.items(), key=lambda x: x[1])
        insights.append(f"📈 **Dominant Category**: {top_category[0]} ({top_category[1]} pages)")
        
        for insight in insights:
            st.markdown(insight)
        
        # Action summary
        st.subheader("🚀 Next Steps")
        
        st.markdown(f"""
        **Priority-Focused Analysis Results for {parsed_url.netloc}:**
        1. **Review Priority Pages Performance** - Check individual priority page rankings
        2. **Optimize Low-Performing Priority Pages** - Increase internal links to underperforming priority pages
        3. **Focus on Target Keywords** - Align internal linking with target keywords
        4. **Reduce Low-Value PageRank** - Limit links to non-commercial content
        5. **Monitor and Track** - Use the Excel report to track improvements
        
        **Your Website Analysis Summary:**
        - **Total Pages Discovered:** {len(page_data)}
        - **Priority Pages Found:** {len(analysis_df[analysis_df['Is_Priority'] == True])}
        - **Categories Discovered:** {len(category_counts)}
        - **High-Value Pages:** {len(analysis_df[analysis_df['Business_Value'] == 'High Business Value'])}
        - **Optimization Opportunities:** {len(recommendations_df)} items identified
        
        **Priority Pages Features:**
        - ✅ Custom column mapping for URLs and keywords
        - ✅ Target keyword performance tracking
        - ✅ Priority vs regular page comparison
        - ✅ Category-specific priority analysis
        
        **Need Help?** The Excel report contains detailed priority page analysis with target keywords and specific recommendations.
        """)

if __name__ == "__main__":
    main()
