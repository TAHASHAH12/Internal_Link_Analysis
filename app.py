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
import plotly.figure_factory as ff
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
import openai
from openai import OpenAI
import os
import threading
from queue import Queue
import hashlib
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell
import tempfile
import gc
import weakref
import pickle
from functools import lru_cache
import logging

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Advanced PageRank SEO Analyzer with AI Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for stunning visuals
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6B73FF 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="40" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="80" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
        opacity: 0.3;
    }
    
    .main-header > * {
        position: relative;
        z-index: 2;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #e8f4fd 0%, #dbeafe 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        transition: transform 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateX(8px);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 20%, #f59e0b 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid #f59e0b;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.25);
        color: #92400e;
    }
    
    .critical-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 20%, #ef4444 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid #ef4444;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.25);
        color: #991b1b;
    }
    
    .success-card {
        background: linear-gradient(135deg, #dcfce7 0%, #86efac 20%, #22c55e 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid #22c55e;
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.25);
        color: #14532d;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #f3e8ff 0%, #c084fc 20%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border-left: 6px solid #8b5cf6;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.25);
        color: #581c87;
    }
    
    .route-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #6ee7b7 20%, #10b981 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
        transition: all 0.3s ease;
    }
    
    .route-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    .question-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.3em;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .question-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 65px;
        padding: 0 32px;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.95);
        color: #667eea;
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.3);
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .route-tree {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .route-node {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .route-node:hover {
        transform: translateX(4px);
    }
    
    .route-level-0 { border-left-color: #ef4444; }
    .route-level-1 { border-left-color: #f97316; }
    .route-level-2 { border-left-color: #eab308; }
    .route-level-3 { border-left-color: #22c55e; }
    .route-level-4 { border-left-color: #3b82f6; }
    .route-level-5 { border-left-color: #8b5cf6; }
</style>
""", unsafe_allow_html=True)

class AdvancedCategoryDetector:
    """Enhanced category detector with business value insights"""
    
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
            'finance': ['loans', 'loan', 'mortgage', 'credit', 'banking', 'finance', 'investment', 'insurance', 'savings'],
            'healthcare': ['health', 'medical', 'doctor', 'hospital', 'clinic', 'treatment'],
            'education': ['course', 'education', 'learn', 'training', 'tutorial', 'class'],
            'technology': ['software', 'app', 'tech', 'digital', 'cloud', 'api', 'development'],
            'ecommerce': ['cart', 'checkout', 'payment', 'order', 'shipping', 'delivery'],
            'real_estate': ['property', 'real-estate', 'house', 'apartment', 'rent', 'buy'],
            'automotive': ['car', 'auto', 'vehicle', 'parts', 'repair', 'dealer'],
            'travel': ['travel', 'hotel', 'flight', 'booking', 'destination', 'tour'],
            'food': ['restaurant', 'food', 'menu', 'recipe', 'cooking', 'dining'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'show', 'event'],
            'institution': ['institution', 'bank', 'university', 'school', 'hospital', 'government'],
            'author': ['author', 'writer', 'journalist', 'contributor', 'expert']
        }
        
        # Business value mapping
        self.business_value = {
            'high': ['product', 'service', 'finance', 'ecommerce', 'homepage', 'real_estate', 'automotive', 'institution'],
            'medium': ['content', 'company', 'education', 'technology', 'healthcare', 'author'],
            'low': ['category', 'tag', 'search', 'legal', 'help', 'user', 'contact']
        }

    @lru_cache(maxsize=2000)
    def categorize_url(self, url, title="", meta_desc="", h1=""):
        """Enhanced URL categorization with business value assessment"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Combine text for analysis
        full_text = f"{path} {title[:200]} {meta_desc[:200]} {h1[:100]}".lower()
        
        # Check against business keywords
        category_scores = {}
        for category, keywords in self.business_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                category_scores[category] = score
        
        # URL pattern analysis
        path_segments = [seg for seg in path.split('/') if seg]
        
        if path in ['/', ''] or len(path_segments) == 0:
            return 'homepage'
        
        # Enhanced pattern matching
        if 'tag' in path or 'tags' in path:
            return 'tag'
        if 'category' in path or 'categories' in path:
            return 'category'
        if 'author' in path or 'writer' in path:
            return 'author'
        if 'news' in path or 'blog' in path or 'article' in path:
            return 'content'
        if 'loan' in path or 'finance' in path or 'bank' in path:
            return 'finance'
        if 'institution' in path or 'university' in path:
            return 'institution'
        
        first_segment = path_segments[0] if path_segments else ''
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return first_segment if first_segment else 'other'
    
    def get_business_value(self, category):
        """Get business value for a category"""
        for value_level, categories in self.business_value.items():
            if category in categories:
                return value_level
        return 'medium'

class InternalRouteMapper:
    """Advanced internal route mapping and visualization"""
    
    def __init__(self):
        self.route_tree = {}
        self.page_hierarchy = defaultdict(list)
        self.breadcrumb_paths = {}
        
    def build_route_tree(self, urls, pagerank_scores, section_mapping):
        """Build comprehensive route tree structure"""
        routes = {}
        
        for url in urls:
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            if not path:
                routes['/'] = {
                    'url': url,
                    'level': 0,
                    'segments': [],
                    'pagerank': pagerank_scores.get(url, 0),
                    'section': section_mapping.get(url, 'other'),
                    'children': []
                }
                continue
                
            segments = [seg for seg in path.split('/') if seg]
            
            # Build hierarchical structure
            current_path = ''
            for i, segment in enumerate(segments):
                current_path += '/' + segment
                
                if current_path not in routes:
                    routes[current_path] = {
                        'url': url if i == len(segments) - 1 else '',
                        'level': i + 1,
                        'segments': segments[:i+1],
                        'pagerank': pagerank_scores.get(url, 0) if i == len(segments) - 1 else 0,
                        'section': section_mapping.get(url, 'other') if i == len(segments) - 1 else 'navigation',
                        'children': [],
                        'parent': '/'.join(segments[:i]) if i > 0 else '/'
                    }
        
        # Build parent-child relationships
        for path, data in routes.items():
            if data['level'] > 0:
                parent_path = '/' + '/'.join(data['segments'][:-1]) if len(data['segments']) > 1 else '/'
                if parent_path in routes:
                    routes[parent_path]['children'].append(path)
        
        return routes
    
    def generate_breadcrumb_analysis(self, routes):
        """Generate breadcrumb analysis for SEO insights"""
        breadcrumb_analysis = {}
        
        for path, data in routes.items():
            if data['url']:  # Only for actual pages
                breadcrumbs = []
                current_segments = data['segments']
                
                for i in range(len(current_segments)):
                    segment_path = '/' + '/'.join(current_segments[:i+1])
                    if segment_path in routes:
                        breadcrumbs.append({
                            'segment': current_segments[i],
                            'path': segment_path,
                            'level': i + 1
                        })
                
                breadcrumb_analysis[data['url']] = {
                    'breadcrumbs': breadcrumbs,
                    'depth': len(breadcrumbs),
                    'pagerank': data['pagerank'],
                    'section': data['section']
                }
        
        return breadcrumb_analysis

class AdvancedPageRankAnalyzer:
    """Enhanced PageRank analyzer with AI insights and route mapping"""
    
    def __init__(self, openai_api_key=None):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.graph = nx.DiGraph()
        self.pagerank_scores = {}
        self.page_data = {}
        self.section_mapping = {}
        self.anchor_texts = defaultdict(Counter)
        self.category_detector = AdvancedCategoryDetector()
        self.route_mapper = InternalRouteMapper()
        self.crawl_stats = {
            'pages_crawled': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
    def crawl_website(self, seed_url, max_pages=5000, depth=3, delay=0.1):
        """Enhanced website crawler with route tracking"""
        visited = set()
        to_visit = [seed_url]
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.crawl_stats['start_time'] = datetime.now()
        
        # Enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
            with col2:
                stats_text = st.empty()
            with col3:
                route_preview = st.empty()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        routes_discovered = set()
        
        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            
            if current_url in visited:
                continue
                
            try:
                # Update progress with enhanced visuals
                progress = len(visited) / max_pages
                progress_bar.progress(min(progress, 1.0))
                
                elapsed_time = (datetime.now() - self.crawl_stats['start_time']).total_seconds()
                pages_per_second = len(visited) / elapsed_time if elapsed_time > 0 else 0
                
                status_text.markdown(f"""
                **🕷️ Crawling Progress**
                - **Current:** `{current_url[:50]}...`
                - **Progress:** {len(visited)}/{max_pages} pages
                - **Queue:** {len(to_visit)} pages
                """)
                
                stats_text.markdown(f"""
                **📊 Live Statistics**
                - **Speed:** {pages_per_second:.1f} pages/sec
                - **Errors:** {self.crawl_stats['errors']}
                - **Time:** {elapsed_time:.0f}s
                """)
                
                # Track routes discovered
                route = urlparse(current_url).path
                if route not in routes_discovered:
                    routes_discovered.add(route)
                    
                route_preview.markdown(f"""
                **🗺️ Routes Discovered**
                - **Total Routes:** {len(routes_discovered)}
                - **Latest:** `{route}`
                - **Depth:** {len([seg for seg in route.split('/') if seg])}
                """)
                
                # Make request
                response = session.get(current_url, timeout=10)
                
                if 'text/html' not in response.headers.get('content-type', '').lower():
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract comprehensive page data
                title = soup.title.string.strip() if soup.title else ''
                h1 = soup.h1.get_text().strip() if soup.h1 else ''
                
                # Extract meta description
                meta_desc = ''
                meta_tag = soup.find('meta', attrs={'name': 'description'})
                if meta_tag:
                    meta_desc = meta_tag.get('content', '')
                
                # Extract structured data
                structured_data = []
                for script in soup.find_all('script', type='application/ld+json'):
                    try:
                        data = json.loads(script.string)
                        structured_data.append(data)
                    except:
                        continue
                
                self.page_data[current_url] = {
                    'title': title,
                    'h1': h1,
                    'meta_description': meta_desc,
                    'word_count': len(soup.get_text().split()),
                    'status_code': response.status_code,
                    'internal_links': 0,
                    'external_links': 0,
                    'outbound_pages': [],
                    'structured_data': structured_data,
                    'route_depth': len([seg for seg in urlparse(current_url).path.split('/') if seg])
                }
                
                # Extract links with detailed analysis
                internal_links = 0
                external_links = 0
                new_urls = []
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '').strip()
                    if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                        continue
                    
                    try:
                        full_url = urljoin(current_url, href).split('#')[0]
                        
                        if self._is_internal_link(full_url, seed_url):
                            internal_links += 1
                            self.graph.add_edge(current_url, full_url)
                            
                            # Store outbound page relationship
                            self.page_data[current_url]['outbound_pages'].append(full_url)
                            
                            # Extract and analyze anchor text
                            anchor_text = link.get_text().strip()
                            if anchor_text:
                                self.anchor_texts[full_url][anchor_text] += 1
                            
                            # Add to crawl queue
                            if (full_url not in visited and 
                                full_url not in to_visit and 
                                full_url not in new_urls and
                                len(visited) + len(to_visit) < max_pages):
                                new_urls.append(full_url)
                        else:
                            external_links += 1
                            
                    except Exception:
                        continue
                
                to_visit.extend(new_urls[:30])
                
                self.page_data[current_url]['internal_links'] = internal_links
                self.page_data[current_url]['external_links'] = external_links
                
                visited.add(current_url)
                self.crawl_stats['pages_crawled'] = len(visited)
                
                time.sleep(delay)
                
            except Exception as e:
                self.crawl_stats['errors'] += 1
                continue
        
        progress_bar.progress(1.0)
        self.crawl_stats['end_time'] = datetime.now()
        
        total_time = (self.crawl_stats['end_time'] - self.crawl_stats['start_time']).total_seconds()
        status_text.markdown(f"""
        **✅ Crawling Completed!**
        - **Total Pages:** {len(visited)}
        - **Total Routes:** {len(routes_discovered)}
        - **Total Links:** {len(self.graph.edges())}
        - **Time:** {total_time:.1f} seconds
        """)
        
        return visited
    
    def _is_internal_link(self, url, seed_url):
        """Check if URL is internal"""
        try:
            return urlparse(url).netloc == urlparse(seed_url).netloc
        except:
            return False
    
    def calculate_pagerank(self, alpha=0.85, max_iter=100):
        """Calculate PageRank with enhanced analytics"""
        if len(self.graph.nodes()) == 0:
            st.error("No pages to analyze")
            return {}
        
        with st.spinner("🧮 Calculating PageRank scores..."):
            try:
                self.pagerank_scores = nx.pagerank(
                    self.graph, 
                    alpha=alpha, 
                    max_iter=max_iter,
                    tol=1e-6
                )
                
                # Normalize scores
                total_score = sum(self.pagerank_scores.values())
                if total_score > 0:
                    self.pagerank_scores = {
                        url: score/total_score 
                        for url, score in self.pagerank_scores.items()
                    }
                
                st.success(f"✅ PageRank calculated for {len(self.pagerank_scores)} pages")
                
            except Exception as e:
                st.error(f"Error calculating PageRank: {str(e)}")
                num_pages = len(self.graph.nodes())
                self.pagerank_scores = {node: 1.0/num_pages for node in self.graph.nodes()}
        
        return self.pagerank_scores
    
    def detect_sections(self, urls):
        """Enhanced section detection"""
        section_patterns = {}
        
        for url in urls:
            page_info = self.page_data.get(url, {})
            title = page_info.get('title', '')
            meta_desc = page_info.get('meta_description', '')
            h1 = page_info.get('h1', '')
            
            category = self.category_detector.categorize_url(url, title, meta_desc, h1)
            section_patterns[url] = category
        
        return section_patterns
    
    def generate_ai_recommendations(self, analysis_data):
        """Generate comprehensive AI-powered recommendations"""
        if not self.openai_client:
            return "OpenAI API key not provided. Please add your API key for AI recommendations."
        
        try:
            prompt = f"""
            As a senior SEO strategist, analyze this comprehensive PageRank data and provide strategic recommendations:
            
            **Website Overview:**
            - Total pages analyzed: {analysis_data.get('total_pages', 0)}
            - Total internal links: {analysis_data.get('total_links', 0)}
            - Sections identified: {len(analysis_data.get('sections', []))}
            
            **PageRank Distribution:**
            - Top sections: {analysis_data.get('top_sections', [])[:5]}
            - Top pages: {len(analysis_data.get('top_pages', []))} analyzed
            - Business value distribution: {analysis_data.get('business_distribution', {})}
            
            **Key Issues Identified:**
            - PageRank waste: {analysis_data.get('waste_percentage', 0):.1f}% in low-value sections
            - Priority page alignment: {analysis_data.get('priority_alignment', 'Not provided')}
            - Internal linking opportunities: {len(analysis_data.get('linking_opportunities', []))}
            
            **Route Structure:**
            - Route depth analysis: {analysis_data.get('route_depth_stats', {})}
            - Navigation efficiency: {analysis_data.get('navigation_efficiency', 'Unknown')}
            
            Please provide:
            
            1. **Immediate Action Items** (0-2 weeks):
               - Specific high-impact changes
               - Quick wins for PageRank redistribution
               - Critical technical fixes
            
            2. **Strategic Optimizations** (1-3 months):
               - Internal linking strategy overhaul
               - Content architecture improvements
               - Section priority rebalancing
            
            3. **Long-term Vision** (3-12 months):
               - Site structure evolution
               - Authority flow optimization
               - Competitive advantage development
            
            4. **Technical Implementation**:
               - Specific code/CMS changes needed
               - Tools and monitoring setup
               - Success metrics to track
            
            5. **Expected Impact Quantification**:
               - Projected PageRank improvements
               - SEO performance uplift estimates
               - Business value creation potential
            
            Make recommendations specific, actionable, and prioritized by impact vs. effort.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a world-class SEO strategist with deep expertise in technical SEO, internal linking, and PageRank optimization. Provide detailed, actionable, and prioritized recommendations that drive real business results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating AI recommendations: {str(e)}"

def create_stunning_visualizations(analyzer, insights):
    """Create beautiful, interactive visualizations"""
    
    # 1. Advanced Section PageRank Sunburst
    section_data = []
    for url, section in analyzer.section_mapping.items():
        pr_score = analyzer.pagerank_scores.get(url, 0)
        business_value = analyzer.category_detector.get_business_value(section)
        
        section_data.append({
            'section': section,
            'url': url,
            'pagerank': pr_score,
            'business_value': business_value,
            'route_depth': analyzer.page_data.get(url, {}).get('route_depth', 0)
        })
    
    df_sections = pd.DataFrame(section_data)
    
    # Sunburst chart for hierarchical view
    fig_sunburst = px.sunburst(
        df_sections.groupby(['business_value', 'section']).agg({
            'pagerank': 'sum',
            'url': 'count'
        }).reset_index(),
        path=['business_value', 'section'],
        values='pagerank',
        color='pagerank',
        color_continuous_scale='RdYlBu_r',
        title='🌅 PageRank Distribution Hierarchy (Business Value → Sections)'
    )
    
    fig_sunburst.update_layout(
        height=600,
        font=dict(size=14),
        title_font_size=20
    )
    
    # 2. 3D Network Visualization
    top_pages = sorted(analyzer.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    top_urls = [url for url, _ in top_pages]
    
    subgraph = analyzer.graph.subgraph(top_urls)
    
    if len(subgraph.nodes()) > 0:
        # Use 3D spring layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50, dim=3)
        
        # Extract coordinates
        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        node_z = [pos[node][2] for node in subgraph.nodes()]
        
        # Create edge traces
        edge_traces = []
        for edge in subgraph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_traces.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(color='rgba(125, 125, 125, 0.4)', width=2),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create node trace
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node in subgraph.nodes():
            pr_score = analyzer.pagerank_scores.get(node, 0)
            section = analyzer.section_mapping.get(node, 'other')
            business_value = analyzer.category_detector.get_business_value(section)
            
            # Color by business value
            if business_value == 'high':
                node_colors.append('#22c55e')
            elif business_value == 'medium':
                node_colors.append('#f59e0b')
            else:
                node_colors.append('#ef4444')
            
            node_sizes.append(max(8, pr_score * 1000))
            
            title = analyzer.page_data.get(node, {}).get('title', '')
            node_texts.append(f"{title[:30]}...<br>PR: {pr_score:.4f}<br>Section: {section}")
        
        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=node_texts,
            hoverinfo='text',
            name='Pages'
        )
        
        fig_3d_network = go.Figure(data=[node_trace] + edge_traces)
        fig_3d_network.update_layout(
            title='🌐 3D PageRank Flow Network (Green=High Value, Yellow=Medium, Red=Low)',
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=700
        )
    else:
        fig_3d_network = go.Figure()
    
    # 3. Advanced Sankey with Business Value Flow
    section_links = defaultdict(lambda: defaultdict(int))
    for source, target in analyzer.graph.edges():
        source_section = analyzer.section_mapping.get(source, 'other')
        target_section = analyzer.section_mapping.get(target, 'other')
        section_links[source_section][target_section] += 1
    
    # Prepare Sankey data with business value
    sections = list(set(analyzer.section_mapping.values()))
    node_labels = []
    node_colors = []
    
    for section in sections:
        business_value = analyzer.category_detector.get_business_value(section)
        section_pr = sum(analyzer.pagerank_scores.get(url, 0) for url, sec in analyzer.section_mapping.items() if sec == section)
        
        node_labels.append(f"{section.title()}<br>PR: {section_pr:.3f}<br>Value: {business_value.title()}")
        
        if business_value == 'high':
            node_colors.append('rgba(34, 197, 94, 0.8)')
        elif business_value == 'medium':
            node_colors.append('rgba(245, 158, 11, 0.8)')
        else:
            node_colors.append('rgba(239, 68, 68, 0.8)')
    
    # Create links
    source_indices = []
    target_indices = []
    link_values = []
    link_colors = []
    
    for source_section, targets in section_links.items():
        if source_section in sections:
            source_idx = sections.index(source_section)
            source_value = analyzer.category_detector.get_business_value(source_section)
            
            for target_section, count in targets.items():
                if source_section != target_section and target_section in sections and count > 0:
                    target_idx = sections.index(target_section)
                    target_value = analyzer.category_detector.get_business_value(target_section)
                    
                    source_indices.append(source_idx)
                    target_indices.append(target_idx)
                    link_values.append(count)
                    
                    # Color links based on business value flow
                    if source_value == 'low' and target_value == 'high':
                        link_colors.append('rgba(34, 197, 94, 0.6)')  # Good flow (green)
                    elif source_value == 'high' and target_value == 'low':
                        link_colors.append('rgba(239, 68, 68, 0.6)')  # Bad flow (red)
                    elif source_value == 'medium':
                        link_colors.append('rgba(245, 158, 11, 0.4)')  # Neutral (yellow)
                    else:
                        link_colors.append('rgba(148, 163, 184, 0.4)')  # Other (gray)
    
    if source_indices:
        fig_sankey = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=link_values,
                color=link_colors
            )
        )])
        
        fig_sankey.update_layout(
            title="🔄 Advanced Section Flow Analysis (Green=Good Flow, Red=PageRank Waste)",
            height=700,
            font_size=12
        )
    else:
        fig_sankey = go.Figure()
    
    # 4. Route Depth Analysis
    route_depths = defaultdict(list)
    for url, data in analyzer.page_data.items():
        depth = data.get('route_depth', 0)
        pr_score = analyzer.pagerank_scores.get(url, 0)
        route_depths[depth].append(pr_score)
    
    depth_analysis = []
    for depth, pr_scores in route_depths.items():
        depth_analysis.append({
            'depth': depth,
            'avg_pagerank': np.mean(pr_scores),
            'total_pagerank': sum(pr_scores),
            'page_count': len(pr_scores),
            'max_pagerank': max(pr_scores) if pr_scores else 0
        })
    
    df_depth = pd.DataFrame(depth_analysis)
    
    fig_depth = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PageRank by Route Depth', 'Page Count by Depth', 
                       'Total PageRank by Depth', 'Max PageRank by Depth'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig_depth.add_trace(
        go.Scatter(x=df_depth['depth'], y=df_depth['avg_pagerank'], 
                  mode='lines+markers', name='Avg PageRank',
                  line=dict(color='#3b82f6', width=3)),
        row=1, col=1
    )
    
    fig_depth.add_trace(
        go.Bar(x=df_depth['depth'], y=df_depth['page_count'], 
               name='Page Count', marker_color='#10b981'),
        row=1, col=2
    )
    
    fig_depth.add_trace(
        go.Scatter(x=df_depth['depth'], y=df_depth['total_pagerank'], 
                  mode='lines+markers', name='Total PageRank',
                  line=dict(color='#f59e0b', width=3)),
        row=2, col=1
    )
    
    fig_depth.add_trace(
        go.Scatter(x=df_depth['depth'], y=df_depth['max_pagerank'], 
                  mode='lines+markers', name='Max PageRank',
                  line=dict(color='#ef4444', width=3)),
        row=2, col=2
    )
    
    fig_depth.update_layout(
        title_text="📊 Route Depth Analysis Dashboard",
        height=600,
        showlegend=False
    )
    
    return {
        'sunburst': fig_sunburst,
        'network_3d': fig_3d_network,
        'sankey': fig_sankey,
        'depth_analysis': fig_depth
    }

def create_route_visualization(analyzer):
    """Create comprehensive internal route visualization"""
    
    # Build route tree
    routes = analyzer.route_mapper.build_route_tree(
        list(analyzer.pagerank_scores.keys()),
        analyzer.pagerank_scores,
        analyzer.section_mapping
    )
    
    # Create hierarchical route display
    st.markdown("### 🗺️ Complete Internal Route Map")
    
    # Group routes by depth
    routes_by_depth = defaultdict(list)
    for path, data in routes.items():
        routes_by_depth[data['level']].append((path, data))
    
    # Display routes in collapsible sections
    for depth in sorted(routes_by_depth.keys()):
        routes_at_depth = routes_by_depth[depth]
        
        with st.expander(f"📁 Level {depth} Routes ({len(routes_at_depth)} routes)", expanded=(depth <= 2)):
            
            # Sort by PageRank for this depth
            routes_at_depth.sort(key=lambda x: x[1]['pagerank'], reverse=True)
            
            cols = st.columns(min(3, len(routes_at_depth)))
            
            for i, (path, data) in enumerate(routes_at_depth):
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    # Determine card color based on business value
                    if data['url']:
                        business_value = analyzer.category_detector.get_business_value(data['section'])
                        if business_value == 'high':
                            card_class = "success-card"
                        elif business_value == 'medium':
                            card_class = "warning-card"
                        else:
                            card_class = "critical-card"
                    else:
                        card_class = "insight-card"
                    
                    # Get page title if available
                    title = "Navigation Path"
                    if data['url']:
                        page_data = analyzer.page_data.get(data['url'], {})
                        title = page_data.get('title', 'Untitled Page')[:50]
                        if len(page_data.get('title', '')) > 50:
                            title += "..."
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>🔗 {title}</h4>
                        <p><strong>Route:</strong> <code>{path}</code></p>
                        <p><strong>Level:</strong> {data['level']}</p>
                        <p><strong>Section:</strong> {data['section']}</p>
                        <p><strong>PageRank:</strong> {data['pagerank']:.6f}</p>
                        {f"<p><strong>URL:</strong> <code>{data['url'][:60]}...</code></p>" if data['url'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Route statistics
    st.markdown("### 📊 Route Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_routes = len(routes)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🗂️ Total Routes</h3>
            <h2>{total_routes}</h2>
            <p>Unique paths discovered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_depth = max(data['level'] for data in routes.values()) if routes else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📏 Max Depth</h3>
            <h2>{max_depth}</h2>
            <p>Deepest route level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_depth = np.mean([data['level'] for data in routes.values()]) if routes else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Avg Depth</h3>
            <h2>{avg_depth:.1f}</h2>
            <p>Average route level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        orphaned_routes = sum(1 for data in routes.values() if data['url'] and data['pagerank'] == 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏝️ Orphaned</h3>
            <h2>{orphaned_routes}</h2>
            <p>Routes with no PageRank</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive route tree visualization
    st.markdown("### 🌳 Interactive Route Tree")
    
    # Create tree structure for visualization
    tree_data = []
    for path, data in routes.items():
        if data['level'] <= 3:  # Limit depth for visualization
            tree_data.append({
                'id': path,
                'parent': data.get('parent', ''),
                'value': data['pagerank'] * 1000 if data['pagerank'] > 0 else 1,
                'label': f"{path.split('/')[-1] or 'Home'}<br>PR: {data['pagerank']:.4f}"
            })
    
    if tree_data:
        fig_treemap = go.Figure(go.Treemap(
            ids=[item['id'] for item in tree_data],
            labels=[item['label'] for item in tree_data],
            parents=[item['parent'] for item in tree_data],
            values=[item['value'] for item in tree_data],
            textinfo="label",
            pathbar_thickness=20,
            maxdepth=4
        ))
        
        fig_treemap.update_layout(
            title="🗺️ Route Structure Treemap (Size = PageRank)",
            height=600
        )
        
        st.plotly_chart(fig_treemap, use_container_width=True)

def handle_priority_pages_upload():
    """Enhanced priority pages upload with validation"""
    st.markdown("### 📁 Priority Pages Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload Priority Pages CSV",
        type=['csv'],
        help="Upload CSV file containing your priority pages and target keywords"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show file preview with enhanced styling
            st.markdown("**📊 File Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection interface
            st.markdown("**🔧 Column Mapping:**")
            col1, col2 = st.columns(2)
            
            with col1:
                url_column = st.selectbox(
                    "Select URL Column",
                    options=df.columns.tolist(),
                    help="Column containing the URLs of your priority pages"
                )
            
            with col2:
                keyword_column = st.selectbox(
                    "Select Keywords Column (Optional)",
                    options=['None'] + df.columns.tolist(),
                    help="Column containing target keywords (comma-separated)"
                )
            
            # Create standardized dataframe
            if url_column:
                priority_df = pd.DataFrame({
                    'URL': df[url_column]
                })
                
                if keyword_column != 'None':
                    priority_df['Target Keywords'] = df[keyword_column]
                else:
                    priority_df['Target Keywords'] = ''
                
                # Validate URLs
                valid_urls = 0
                for url in priority_df['URL']:
                    try:
                        parsed = urlparse(str(url))
                        if parsed.netloc:
                            valid_urls += 1
                    except:
                        continue
                
                st.markdown(f"""
                <div class="success-card">
                    <h4>✅ Configuration Complete</h4>
                    <p><strong>Total Pages:</strong> {len(priority_df)}</p>
                    <p><strong>Valid URLs:</strong> {valid_urls}</p>
                    <p><strong>Success Rate:</strong> {(valid_urls/len(priority_df)*100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                return priority_df
            
        except Exception as e:
            st.markdown(f"""
            <div class="critical-card">
                <h4>❌ Error Reading File</h4>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
            return None
    
    return None

def main():
    # Stunning header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced PageRank SEO Analyzer</h1>
        <h2>AI-Powered Internal Linking Analysis with Route Mapping</h2>
        <p>Comprehensive analysis with stunning visualizations and intelligent recommendations</p>
        <p><strong>✨ Now with OpenAI insights and complete internal route visualization ✨</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("### 🔧 Configuration Panel")
    
    # OpenAI API Key with enhanced styling
    openai_key = st.sidebar.text_input(
        "🤖 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key for AI-powered recommendations"
    )
    
    if openai_key:
        st.sidebar.markdown("✅ **AI Recommendations Enabled**")
    else:
        st.sidebar.markdown("⚠️ **Add API key for AI insights**")
    
    # Website URL
    website_url = st.sidebar.text_input(
        "🌐 Website URL",
        placeholder="https://example.com",
        help="Enter the website you want to analyze"
    )
    
    # Advanced crawling parameters
    st.sidebar.markdown("### ⚙️ Crawling Settings")
    
    max_pages = st.sidebar.slider("📄 Max Pages to Crawl", 100, 5000, 1000, step=100)
    crawl_depth = st.sidebar.slider("🕳️ Crawl Depth", 1, 5, 3)
    crawl_delay = st.sidebar.slider("⏱️ Crawl Delay (seconds)", 0.1, 2.0, 0.2, step=0.1)
    
    # Priority Pages Upload
    priority_pages_df = handle_priority_pages_upload()
    
    # Enhanced Analysis Button
    if st.sidebar.button("🚀 Start Advanced Analysis", type="primary"):
        if not website_url:
            st.markdown("""
            <div class="critical-card">
                <h4>❌ Missing Website URL</h4>
                <p>Please enter a website URL to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Initialize analyzer
        analyzer = AdvancedPageRankAnalyzer(openai_key)
        
        # Crawl website with enhanced progress tracking
        st.markdown("## 🕷️ Website Crawling & Data Collection")
        crawled_urls = analyzer.crawl_website(website_url, max_pages, crawl_depth, crawl_delay)
        
        if not crawled_urls:
            st.markdown("""
            <div class="critical-card">
                <h4>❌ No Pages Found</h4>
                <p>Please check the URL and try again.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        st.markdown(f"""
        <div class="success-card">
            <h4>✅ Crawling Successful!</h4>
            <p><strong>Pages Discovered:</strong> {len(crawled_urls)}</p>
            <p><strong>Internal Links:</strong> {len(analyzer.graph.edges())}</p>
            <p><strong>Unique Routes:</strong> {len(set(urlparse(url).path for url in crawled_urls))}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detect sections
        st.info("🔍 Analyzing website structure and categorizing pages...")
        analyzer.section_mapping = analyzer.detect_sections(list(crawled_urls))
        
        # Calculate PageRank
        st.markdown("## 📊 PageRank Calculation & Analysis")
        pagerank_scores = analyzer.calculate_pagerank()
        
        if not pagerank_scores:
            st.error("❌ Could not calculate PageRank scores.")
            return
        
        # Create visualizations
        st.info("🎨 Creating stunning visualizations...")
        visualizations = create_stunning_visualizations(analyzer, {})
        
        # Display results in enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 5 Key Questions", "🗺️ Internal Routes", "📊 Advanced Visuals", 
            "🔗 Section Analysis", "🤖 AI Recommendations", "📈 Performance Insights"
        ])
        
        with tab1:
            st.markdown("## 🎯 Analysis of the 5 Critical Questions")
            
            # Question 1: Which sections receive most PR?
            st.markdown('<div class="question-header">1. Which Sections of the site are receiving the most PageRank?</div>', unsafe_allow_html=True)
            
            # Calculate section PageRank
            section_pr = defaultdict(float)
            for url, score in pagerank_scores.items():
                section = analyzer.section_mapping.get(url, 'other')
                section_pr[section] += score
            
            top_sections = sorted(section_pr.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create enhanced section chart
                section_df = pd.DataFrame([
                    {
                        'section': section,
                        'pagerank': pr_score,
                        'percentage': (pr_score / sum(section_pr.values()) * 100) if section_pr else 0,
                        'business_value': analyzer.category_detector.get_business_value(section)
                    }
                    for section, pr_score in top_sections
                ])
                
                fig_sections = px.bar(
                    section_df.head(10),
                    x='section',
                    y='pagerank',
                    color='business_value',
                    color_discrete_map={
                        'high': '#22c55e',
                        'medium': '#f59e0b',
                        'low': '#ef4444'
                    },
                    title='📊 Section PageRank Distribution',
                    labels={'pagerank': 'PageRank Score', 'section': 'Section'}
                )
                
                fig_sections.update_layout(height=500)
                st.plotly_chart(fig_sections, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="insight-card">
                    <h4>🏆 Top Sections</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for i, (section, score) in enumerate(top_sections[:5], 1):
                    percentage = (score / sum(section_pr.values()) * 100) if section_pr else 0
                    business_value = analyzer.category_detector.get_business_value(section)
                    value_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[business_value]
                    
                    st.markdown(f"""
                    **{i}. {section.title()}** {value_emoji}  
                    📊 {score:.4f} PR ({percentage:.1f}%)  
                    📄 {sum(1 for url, sec in analyzer.section_mapping.items() if sec == section)} pages  
                    💼 {business_value.title()} business value
                    """)
            
            # Question 2: Which specific pages receive most PR?
            st.markdown('<div class="question-header">2. Which specific pages receive the most PageRank?</div>', unsafe_allow_html=True)
            
            top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]
            
            top_pages_data = []
            for i, (url, score) in enumerate(top_pages, 1):
                page_info = analyzer.page_data.get(url, {})
                section = analyzer.section_mapping.get(url, 'other')
                business_value = analyzer.category_detector.get_business_value(section)
                
                top_pages_data.append({
                    'Rank': i,
                    'URL': url[:80] + '...' if len(url) > 80 else url,
                    'PageRank Score': f"{score:.6f}",
                    'Section': section,
                    'Business Value': business_value,
                    'Title': page_info.get('title', '')[:50] + '...' if len(page_info.get('title', '')) > 50 else page_info.get('title', ''),
                    'Route Depth': page_info.get('route_depth', 0),
                    'Internal Links': page_info.get('internal_links', 0)
                })
            
            st.dataframe(pd.DataFrame(top_pages_data), use_container_width=True, height=600)
            
            # Question 3: Priority pages alignment
            st.markdown('<div class="question-header">3. Do these align with your Priority Target Pages?</div>', unsafe_allow_html=True)
            
            if priority_pages_df is not None and not priority_pages_df.empty:
                priority_analysis = []
                all_scores = sorted(pagerank_scores.values(), reverse=True)
                
                for _, row in priority_pages_df.iterrows():
                    url = row.get('URL', '')
                    if url in pagerank_scores:
                        score = pagerank_scores[url]
                        rank = all_scores.index(score) + 1
                        percentile = (rank / len(pagerank_scores)) * 100
                        
                        priority_analysis.append({
                            'url': url,
                            'pagerank': score,
                            'rank': rank,
                            'percentile': percentile,
                            'section': analyzer.section_mapping.get(url, 'other')
                        })
                
                if priority_analysis:
                    avg_percentile = np.mean([p['percentile'] for p in priority_analysis])
                    alignment_score = max(0, 100 - avg_percentile)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        color = 'success' if alignment_score > 70 else 'warning' if alignment_score > 40 else 'critical'
                        st.markdown(f"""
                        <div class="{color}-card">
                            <h3>🎯 Alignment Score</h3>
                            <h2>{alignment_score:.1f}/100</h2>
                            <p>Higher = Better alignment with PageRank distribution</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        found_count = len(priority_analysis)
                        total_count = len(priority_pages_df)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 Coverage</h3>
                            <h2>{found_count}/{total_count}</h2>
                            <p>{(found_count/total_count*100):.1f}% of priority pages found</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        avg_rank = np.mean([p['rank'] for p in priority_analysis])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Avg Rank</h3>
                            <h2>{avg_rank:.0f}</h2>
                            <p>Lower ranks = better performance</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Priority pages performance visualization
                    priority_df = pd.DataFrame(priority_analysis)
                    
                    fig_priority = px.scatter(
                        priority_df,
                        x='rank',
                        y='pagerank',
                        size='pagerank',
                        color='section',
                        title='🎯 Priority Pages Performance Analysis',
                        labels={'rank': 'PageRank Rank', 'pagerank': 'PageRank Score'}
                    )
                    
                    fig_priority.update_layout(height=500)
                    st.plotly_chart(fig_priority, use_container_width=True)
                    
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <h4>⚠️ No Priority Pages Found</h4>
                        <p>None of your priority pages were found in the crawled data. They may be too deep or not linked properly.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-card">
                    <h4>📁 Upload Priority Pages</h4>
                    <p>Upload a CSV file with your priority pages to analyze alignment with PageRank distribution.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Question 4: PR waste analysis
            st.markdown('<div class="question-header">4. How can we reduce the PageRank to non-valuable sections?</div>', unsafe_allow_html=True)
            
            # Calculate waste
            low_value_sections = [section for section in section_pr.keys() 
                                if analyzer.category_detector.get_business_value(section) == 'low']
            wasted_pr = sum(section_pr[section] for section in low_value_sections)
            total_pr = sum(section_pr.values())
            waste_percentage = (wasted_pr / total_pr * 100) if total_pr > 0 else 0
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                color = 'critical' if waste_percentage > 20 else 'warning' if waste_percentage > 10 else 'success'
                st.markdown(f"""
                <div class="{color}-card">
                    <h3>🔴 PageRank Waste</h3>
                    <h2>{waste_percentage:.1f}%</h2>
                    <p>Of total PageRank going to low-value sections</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Business value pie chart
                business_values = {'high': 0, 'medium': 0, 'low': 0}
                for section, pr_score in section_pr.items():
                    bv = analyzer.category_detector.get_business_value(section)
                    business_values[bv] += pr_score
                
                fig_bv_pie = px.pie(
                    values=list(business_values.values()),
                    names=['High Value', 'Medium Value', 'Low Value'],
                    color_discrete_map={
                        'High Value': '#22c55e',
                        'Medium Value': '#f59e0b',
                        'Low Value': '#ef4444'
                    },
                    title='💼 PageRank by Business Value'
                )
                
                st.plotly_chart(fig_bv_pie, use_container_width=True)
            
            # Question 5: PR redistribution strategy
            st.markdown('<div class="question-header">5. How can we redirect this PageRank to priority pages?</div>', unsafe_allow_html=True)
            
            redistribution_strategies = []
            
            # Find high-PR low-value pages
            low_value_high_pr_pages = []
            for url, score in pagerank_scores.items():
                section = analyzer.section_mapping.get(url, 'other')
                if analyzer.category_detector.get_business_value(section) == 'low' and score > 0.01:
                    low_value_high_pr_pages.append((url, score, section))
            
            if low_value_high_pr_pages:
                redistribution_strategies.append({
                    'title': 'Reduce Links to Low-Value High-PR Pages',
                    'description': f'Found {len(low_value_high_pr_pages)} low-value pages with significant PageRank',
                    'impact': 'High',
                    'pages': low_value_high_pr_pages[:5]
                })
            
            # Find sections that over-link to low-value areas
            section_linking = defaultdict(lambda: defaultdict(int))
            for source, target in analyzer.graph.edges():
                source_section = analyzer.section_mapping.get(source, 'other')
                target_section = analyzer.section_mapping.get(target, 'other')
                section_linking[source_section][target_section] += 1
            
            for source_section, targets in section_linking.items():
                total_links = sum(targets.values())
                low_value_links = sum(count for target_section, count in targets.items() 
                                    if analyzer.category_detector.get_business_value(target_section) == 'low')
                
                if total_links > 0 and (low_value_links / total_links) > 0.3:
                    redistribution_strategies.append({
                        'title': f'Optimize {source_section.title()} Section Linking',
                        'description': f'{(low_value_links/total_links*100):.1f}% of links go to low-value sections',
                        'impact': 'Medium',
                        'action': f'Reduce low-value links and add priority page links'
                    })
            
            for i, strategy in enumerate(redistribution_strategies, 1):
                st.markdown(f"""
                <div class="insight-card">
                    <h4>{i}. {strategy['title']} ({strategy['impact']} Impact)</h4>
                    <p>{strategy['description']}</p>
                    {f"<p><strong>Action:</strong> {strategy.get('action', 'Review and optimize link allocation')}</p>" if 'action' in strategy else ""}
                </div>
                """, unsafe_allow_html=True)
                
                if 'pages' in strategy:
                    st.markdown("**Top pages to review:**")
                    for url, score, section in strategy['pages']:
                        st.markdown(f"- `{url[:60]}...` (PR: {score:.4f}, Section: {section})")
        
        with tab2:
            st.markdown("## 🗺️ Complete Internal Route Analysis")
            create_route_visualization(analyzer)
        
        with tab3:
            st.markdown("## 📊 Advanced Visualizations")
            
            # Sunburst chart
            st.markdown("### 🌅 Hierarchical PageRank Distribution")
            st.plotly_chart(visualizations['sunburst'], use_container_width=True)
            
            # 3D Network
            st.markdown("### 🌐 3D PageRank Flow Network")
            st.plotly_chart(visualizations['network_3d'], use_container_width=True)
            
            # Enhanced Sankey
            st.markdown("### 🔄 Advanced Section Flow Analysis")
            st.plotly_chart(visualizations['sankey'], use_container_width=True)
            
            # Route depth analysis
            st.markdown("### 📊 Route Depth Performance Dashboard")
            st.plotly_chart(visualizations['depth_analysis'], use_container_width=True)
        
        with tab4:
            st.markdown("## 🔗 Deep Section Analysis")
            
            # Section performance matrix
            st.markdown("### 📊 Section Performance Matrix")
            
            section_matrix_data = []
            for section, pr_score in section_pr.items():
                page_count = sum(1 for url, sec in analyzer.section_mapping.items() if sec == section)
                avg_pr = pr_score / page_count if page_count > 0 else 0
                business_value = analyzer.category_detector.get_business_value(section)
                
                section_matrix_data.append({
                    'Section': section.title(),
                    'Total PageRank': f"{pr_score:.4f}",
                    'Page Count': page_count,
                    'Avg PR/Page': f"{avg_pr:.6f}",
                    'Business Value': business_value.title(),
                    '% of Total PR': f"{(pr_score/total_pr*100):.1f}%" if total_pr > 0 else "0%"
                })
            
            section_matrix_df = pd.DataFrame(section_matrix_data)
            section_matrix_df = section_matrix_df.sort_values('Total PageRank', ascending=False)
            
            st.dataframe(section_matrix_df, use_container_width=True, height=400)
            
            # Section linking analysis
            st.markdown("### 🔗 Section Linking Patterns")
            
            # Create linking matrix
            sections = list(section_pr.keys())
            linking_matrix = pd.DataFrame(index=sections, columns=sections, data=0)
            
            for source_section, targets in section_linking.items():
                for target_section, count in targets.items():
                    if source_section in linking_matrix.index and target_section in linking_matrix.columns:
                        linking_matrix.loc[source_section, target_section] = count
            
            # Create heatmap
            fig_heatmap = px.imshow(
                linking_matrix.values,
                x=linking_matrix.columns,
                y=linking_matrix.index,
                color_continuous_scale='Blues',
                title='🔥 Section-to-Section Linking Heatmap'
            )
            
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab5:
            st.markdown("## 🤖 AI-Powered Strategic Recommendations")
            
            if openai_key:
                if st.button("🧠 Generate Comprehensive AI Analysis", type="primary"):
                    with st.spinner("🤖 AI is analyzing your PageRank data..."):
                        
                        # Prepare comprehensive analysis data
                        analysis_data = {
                            'total_pages': len(crawled_urls),
                            'total_links': len(analyzer.graph.edges()),
                            'sections': list(section_pr.keys()),
                            'top_sections': [(section, score, (score/total_pr*100) if total_pr > 0 else 0) 
                                           for section, score in top_sections[:5]],
                            'top_pages': top_pages[:10],
                            'business_distribution': {
                                bv: sum(section_pr[section] for section in section_pr.keys() 
                                       if analyzer.category_detector.get_business_value(section) == bv)
                                for bv in ['high', 'medium', 'low']
                            },
                            'waste_percentage': waste_percentage,
                            'priority_alignment': alignment_score if 'alignment_score' in locals() else 'Not analyzed',
                            'linking_opportunities': redistribution_strategies,
                            'route_depth_stats': {
                                'max_depth': max(analyzer.page_data.get(url, {}).get('route_depth', 0) for url in crawled_urls),
                                'avg_depth': np.mean([analyzer.page_data.get(url, {}).get('route_depth', 0) for url in crawled_urls])
                            }
                        }
                        
                        ai_recommendations = analyzer.generate_ai_recommendations(analysis_data)
                        
                        st.markdown(f"""
                        <div class="ai-card">
                            <h3>🤖 AI Strategic Analysis & Recommendations</h3>
                            <div style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6;">{ai_recommendations}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>🤖 AI Recommendations Available</h4>
                    <p>Add your OpenAI API key in the sidebar to unlock powerful AI-driven strategic recommendations tailored to your specific PageRank analysis.</p>
                    <p><strong>AI will provide:</strong></p>
                    <ul>
                        <li>🎯 Immediate action items with specific implementation steps</li>
                        <li>📈 Strategic optimizations with timeline recommendations</li>
                        <li>🔮 Long-term vision for PageRank optimization</li>
                        <li>⚡ Technical implementation guidance</li>
                        <li>📊 Expected impact quantification</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Automated insights
            st.markdown("### 🔍 Automated Strategic Insights")
            
            insights = []
            
            # Critical insights
            if waste_percentage > 25:
                insights.append({
                    'type': 'critical',
                    'title': 'Critical PageRank Waste Detected',
                    'description': f'{waste_percentage:.1f}% of PageRank is flowing to low-value sections',
                    'action': 'Immediately audit and reduce internal links to tag, category, and help pages'
                })
            
            # Opportunity insights
            if priority_pages_df is not None and 'alignment_score' in locals() and alignment_score < 50:
                insights.append({
                    'type': 'warning',
                    'title': 'Priority Pages Underperforming',
                    'description': f'Priority pages alignment score: {alignment_score:.1f}/100',
                    'action': 'Increase internal links from high-authority pages to priority pages'
                })
            
            # Positive insights
            high_value_pr = business_values.get('high', 0) / total_pr * 100 if total_pr > 0 else 0
            if high_value_pr > 60:
                insights.append({
                    'type': 'success',
                    'title': 'Strong High-Value PageRank Distribution',
                    'description': f'{high_value_pr:.1f}% of PageRank flows to high-value sections',
                    'action': 'Maintain current strategy and look for incremental optimizations'
                })
            
            for insight in insights:
                card_type = f"{insight['type']}-card"
                st.markdown(f"""
                <div class="{card_type}">
                    <h4>{insight['title']}</h4>
                    <p><strong>Analysis:</strong> {insight['description']}</p>
                    <p><strong>Recommended Action:</strong> {insight['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab6:
            st.markdown("## 📈 Performance Insights & Export")
            
            # Performance summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                efficiency_score = (business_values.get('high', 0) / total_pr * 100) if total_pr > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Efficiency Score</h3>
                    <h2>{efficiency_score:.1f}%</h2>
                    <p>High-value PageRank ratio</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                structural_score = max(0, 100 - (waste_percentage * 2))
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏗️ Structure Score</h3>
                    <h2>{structural_score:.1f}/100</h2>
                    <p>Internal linking quality</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                optimization_potential = min(waste_percentage * 2, 100)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🚀 Optimization Potential</h3>
                    <h2>{optimization_potential:.1f}%</h2>
                    <p>Possible improvement</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if priority_pages_df is not None and 'alignment_score' in locals():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Priority Alignment</h3>
                        <h2>{alignment_score:.1f}/100</h2>
                        <p>Strategic focus score</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Coverage</h3>
                        <h2>{len(crawled_urls)}</h2>
                        <p>Pages analyzed</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Export functionality
            st.markdown("### 📤 Export Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Comprehensive JSON Report", type="primary"):
                    report_data = {
                        'analysis_summary': {
                            'website_url': website_url,
                            'analysis_date': datetime.now().isoformat(),
                            'total_pages': len(crawled_urls),
                            'total_links': len(analyzer.graph.edges()),
                            'efficiency_score': efficiency_score,
                            'waste_percentage': waste_percentage
                        },
                        'section_analysis': {
                            section: {
                                'pagerank': float(pr_score),
                                'percentage': float((pr_score/total_pr*100) if total_pr > 0 else 0),
                                'business_value': analyzer.category_detector.get_business_value(section),
                                'page_count': sum(1 for url, sec in analyzer.section_mapping.items() if sec == section)
                            }
                            for section, pr_score in section_pr.items()
                        },
                        'top_pages': [
                            {
                                'url': url,
                                'pagerank': float(score),
                                'rank': i+1,
                                'section': analyzer.section_mapping.get(url, 'other'),
                                'title': analyzer.page_data.get(url, {}).get('title', '')
                            }
                            for i, (url, score) in enumerate(top_pages[:50])
                        ],
                        'recommendations': redistribution_strategies,
                        'route_analysis': {
                            url: {
                                'route_depth': analyzer.page_data.get(url, {}).get('route_depth', 0),
                                'pagerank': float(analyzer.pagerank_scores.get(url, 0)),
                                'section': analyzer.section_mapping.get(url, 'other')
                            }
                            for url in crawled_urls
                        }
                    }
                    
                    report_json = json.dumps(report_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=report_json,
                        file_name=f"pagerank_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📋 Download CSV Summary", type="secondary"):
                    # Create comprehensive CSV
                    csv_data = []
                    for url in crawled_urls:
                        page_data = analyzer.page_data.get(url, {})
                        csv_data.append({
                            'URL': url,
                            'PageRank_Score': analyzer.pagerank_scores.get(url, 0),
                            'Section': analyzer.section_mapping.get(url, 'other'),
                            'Business_Value': analyzer.category_detector.get_business_value(analyzer.section_mapping.get(url, 'other')),
                            'Route_Depth': page_data.get('route_depth', 0),
                            'Title': page_data.get('title', ''),
                            'Word_Count': page_data.get('word_count', 0),
                            'Internal_Links': page_data.get('internal_links', 0),
                            'External_Links': page_data.get('external_links', 0)
                        })
                    
                    csv_df = pd.DataFrame(csv_data)
                    csv_string = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_string,
                        file_name=f"pagerank_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
