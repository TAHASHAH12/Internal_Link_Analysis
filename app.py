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
import json
import re
from datetime import datetime
from collections import defaultdict, Counter
import warnings
import openai
from openai import OpenAI
import os
import gc
from functools import lru_cache
import logging
import io
import csv

import logging

# Add this line after the existing logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Configure Streamlit
st.set_page_config(
    page_title="🚀 Advanced PageRank SEO Analyzer with Enhanced AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
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
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
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
    
    .question-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        font-size: 1.3em;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
    }
    
    .recommendation-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 2px solid #0ea5e9;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.15);
    }
    
    .priority-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%);
        border-left: 6px solid #ef4444;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
        border-left: 6px solid #f59e0b;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #dcfce7 0%, #86efac 100%);
        border-left: 6px solid #22c55e;
    }
</style>
""", unsafe_allow_html=True)

class OptimizedCrawler:
    """Memory-efficient web crawler with enhanced analytics"""
    
    def __init__(self, seed_url, max_pages=1000, max_depth=3, delay=0.1):
        self.seed_url = seed_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Core data structures
        self.graph = nx.DiGraph()
        self.page_data = {}
        self.anchor_texts = defaultdict(Counter)
        self.crawl_stats = {
            'pages_crawled': 0,
            'links_found': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'domains_found': set(),
            'response_codes': Counter()
        }
    
    def crawl(self):
        """Execute optimized crawling with detailed progress tracking"""
        visited = set()
        to_visit = [(self.seed_url, 0)]
        domain = urlparse(self.seed_url).netloc
        
        self.crawl_stats['start_time'] = datetime.now()
        
        # Enhanced progress tracking
        progress_container = st.container()
        with progress_container:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            with col2:
                stats_text = st.empty()
            
            with col3:
                live_metrics = st.empty()
        
        while to_visit and len(visited) < self.max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited or depth > self.max_depth:
                continue
            
            try:
                # Update progress
                progress = len(visited) / self.max_pages
                progress_bar.progress(min(progress, 1.0))
                
                elapsed_time = (datetime.now() - self.crawl_stats['start_time']).total_seconds()
                pages_per_second = len(visited) / elapsed_time if elapsed_time > 0 else 0
                
                status_text.markdown(f"""
                **🕷️ Crawling Progress**
                - **Current:** `{current_url[:45]}...`
                - **Progress:** {len(visited)}/{self.max_pages}
                - **Depth:** {depth}/{self.max_depth}
                """)
                
                stats_text.markdown(f"""
                **📊 Live Statistics**
                - **Speed:** {pages_per_second:.1f} pages/sec
                - **Links:** {self.crawl_stats['links_found']}
                - **Queue:** {len(to_visit)}
                """)
                
                live_metrics.markdown(f"""
                **⚡ Performance**
                - **Errors:** {self.crawl_stats['errors']}
                - **Time:** {elapsed_time:.1f}s
                - **ETA:** {((self.max_pages - len(visited)) / max(pages_per_second, 0.1)):.0f}s
                """)
                
                # Make request
                response = self.session.get(current_url, timeout=8)
                self.crawl_stats['response_codes'][response.status_code] += 1
                
                if 'text/html' not in response.headers.get('content-type', '').lower():
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract comprehensive page data
                title = soup.title.string.strip() if soup.title else ''
                h1 = soup.h1.get_text().strip() if soup.h1 else ''
                
                meta_desc = ''
                meta_tag = soup.find('meta', attrs={'name': 'description'})
                if meta_tag:
                    meta_desc = meta_tag.get('content', '')
                
                # Enhanced page metrics
                text_content = soup.get_text()
                word_count = len(text_content.split())
                
                self.page_data[current_url] = {
                    'title': title[:200],
                    'h1': h1[:150],
                    'meta_description': meta_desc[:300],
                    'word_count': word_count,
                    'route_depth': len([seg for seg in urlparse(current_url).path.split('/') if seg]),
                    'internal_links': 0,
                    'external_links': 0,
                    'status_code': response.status_code,
                    'content_length': len(response.text),
                    'images': len(soup.find_all('img')),
                    'headers': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
                }
                
                # Extract links with enhanced tracking
                internal_links = 0
                external_links = 0
                new_urls = []
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '').strip()
                    
                    if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                        continue
                    
                    try:
                        full_url = urljoin(current_url, href).split('#')[0]
                        link_domain = urlparse(full_url).netloc
                        
                        if link_domain == domain:
                            internal_links += 1
                            self.graph.add_edge(current_url, full_url)
                            self.crawl_stats['links_found'] += 1
                            
                            # Enhanced anchor text extraction
                            anchor_text = link.get_text().strip()[:150]
                            if anchor_text:
                                self.anchor_texts[full_url][anchor_text] += 1
                            
                            # Add to crawl queue with deduplication
                            if (full_url not in visited and 
                                full_url not in [url for url, _ in to_visit] and
                                full_url not in new_urls and
                                len(visited) + len(to_visit) < self.max_pages):
                                new_urls.append(full_url)
                        else:
                            external_links += 1
                            self.crawl_stats['domains_found'].add(link_domain)
                    except Exception as link_error:
                        continue
                
                # Add new URLs to queue (with intelligent prioritization)
                for new_url in sorted(new_urls, key=lambda x: len(urlparse(x).path))[:20]:
                    to_visit.append((new_url, depth + 1))
                
                self.page_data[current_url]['internal_links'] = internal_links
                self.page_data[current_url]['external_links'] = external_links
                
                visited.add(current_url)
                self.crawl_stats['pages_crawled'] = len(visited)
                
                # Respectful delay
                time.sleep(self.delay)
                
            except Exception as e:
                self.crawl_stats['errors'] += 1
                logger.error(f"Error crawling {current_url}: {str(e)}")
                continue
        
        # Final statistics
        self.crawl_stats['end_time'] = datetime.now()
        total_time = (self.crawl_stats['end_time'] - self.crawl_stats['start_time']).total_seconds()
        
        progress_bar.progress(1.0)
        status_text.markdown(f"""
        **✅ Crawling Complete!**
        - **Pages:** {len(visited)}
        - **Links:** {self.crawl_stats['links_found']}
        - **Time:** {total_time:.1f}s
        """)
        
        return visited

class AdvancedSectionAnalyzer:
    """Enhanced section categorization with business intelligence"""
    
    def __init__(self):
        self.business_keywords = {
            'homepage': ['home', 'index', 'main', 'welcome'],
            'product': ['product', 'products', 'shop', 'store', 'buy', 'purchase', 'item', 'catalog', 'inventory'],
            'service': ['service', 'services', 'solutions', 'consulting', 'support', 'offering'],
            'content': ['blog', 'news', 'article', 'post', 'content', 'insights', 'resources', 'guide'],
            'company': ['about', 'company', 'team', 'history', 'mission', 'vision', 'careers'],
            'contact': ['contact', 'contactus', 'touch', 'location', 'address', 'phone', 'reach'],
            'legal': ['privacy', 'terms', 'legal', 'policy', 'conditions', 'disclaimer', 'gdpr'],
            'help': ['help', 'support', 'faq', 'guide', 'documentation', 'manual', 'tutorial'],
            'user': ['login', 'register', 'signup', 'account', 'profile', 'dashboard', 'member'],
            'category': ['category', 'categories', 'tag', 'tags', 'topic', 'topics', 'archive'],
            'search': ['search', 'results', 'find', 'query', 'filter'],
            'finance': ['loan', 'loans', 'mortgage', 'credit', 'banking', 'finance', 'investment', 'insurance'],
            'healthcare': ['health', 'medical', 'doctor', 'hospital', 'clinic', 'treatment', 'wellness'],
            'education': ['course', 'education', 'learn', 'training', 'tutorial', 'class', 'academy'],
            'technology': ['software', 'app', 'tech', 'digital', 'cloud', 'api', 'development', 'innovation'],
            'ecommerce': ['cart', 'checkout', 'payment', 'order', 'shipping', 'delivery', 'purchase'],
            'real_estate': ['property', 'real-estate', 'house', 'apartment', 'rent', 'buy', 'listing'],
            'automotive': ['car', 'auto', 'vehicle', 'parts', 'repair', 'dealer', 'automotive'],
            'travel': ['travel', 'hotel', 'flight', 'booking', 'destination', 'tour', 'vacation'],
            'food': ['restaurant', 'food', 'menu', 'recipe', 'cooking', 'dining', 'cuisine'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'show', 'event', 'media'],
            'institution': ['institution', 'bank', 'university', 'school', 'hospital', 'government'],
            'author': ['author', 'writer', 'journalist', 'contributor', 'expert', 'profile']
        }
        
        # Enhanced business value mapping with conversion potential
        self.business_value = {
            'critical': ['homepage', 'product', 'service', 'ecommerce'],  # Direct conversion pages
            'high': ['finance', 'institution', 'real_estate', 'automotive', 'healthcare'],  # High-value services
            'medium': ['content', 'company', 'education', 'technology', 'author'],  # Engagement/trust building
            'low': ['category', 'search', 'help', 'user'],  # Utility pages
            'minimal': ['legal', 'contact', 'tag']  # Necessary but low-conversion
        }
        
        # Conversion potential scoring
        self.conversion_scores = {
            'critical': 95,
            'high': 80,
            'medium': 60,
            'low': 35,
            'minimal': 15
        }
    
    @lru_cache(maxsize=1000)
    def categorize_url(self, url, title="", meta_desc="", h1=""):
        """Advanced URL categorization with context analysis"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Combine all textual context
        full_context = f"{path} {title[:200]} {meta_desc[:200]} {h1[:100]}".lower()
        
        # Handle root/homepage
        if path in ['/', ''] or len([seg for seg in path.split('/') if seg]) == 0:
            return 'homepage'
        
        # Enhanced pattern matching with context
        path_segments = [seg for seg in path.split('/') if seg]
        
        # Priority pattern matching
        if any(x in path for x in ['tag', 'tags']) or 'tag=' in parsed.query:
            return 'category'
        if any(x in path for x in ['category', 'categories', 'cat']):
            return 'category'
        if any(x in path for x in ['author', 'writer', 'by']):
            return 'author'
        if any(x in path for x in ['news', 'blog', 'article', 'post']):
            return 'content'
        if any(x in path for x in ['loan', 'finance', 'bank', 'credit']):
            return 'finance'
        if any(x in path for x in ['product', 'shop', 'store', 'buy']):
            return 'product'
        if any(x in path for x in ['service', 'solution']):
            return 'service'
        
        # Context-based keyword scoring
        category_scores = {}
        for category, keywords in self.business_keywords.items():
            score = 0
            for keyword in keywords:
                # Weight different sources differently
                if keyword in path:
                    score += 3  # URL path has highest weight
                if keyword in title.lower():
                    score += 2  # Title is important
                if keyword in h1.lower():
                    score += 2  # H1 is important
                if keyword in meta_desc.lower():
                    score += 1  # Meta description has lower weight
            
            if score > 0:
                category_scores[category] = score
        
        # Return best matching category
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to first path segment
        return path_segments[0] if path_segments else 'other'
    
    def get_business_value(self, category):
        """Get enhanced business value classification"""
        for value_level, categories in self.business_value.items():
            if category in categories:
                return value_level
        return 'medium'
    
    def get_conversion_score(self, category):
        """Get conversion potential score (0-100)"""
        business_value = self.get_business_value(category)
        return self.conversion_scores.get(business_value, 50)

class EnhancedVisualizer:
    """Create comprehensive visualizations for PageRank analysis"""
    
    def __init__(self, pagerank_scores, section_mapping, section_analyzer, graph, page_data):
        self.pagerank_scores = pagerank_scores
        self.section_mapping = section_mapping
        self.section_analyzer = section_analyzer
        self.graph = graph
        self.page_data = page_data
    
    def create_section_bar_chart(self):
        """Enhanced section PageRank bar chart with business value"""
        section_pr = defaultdict(float)
        section_counts = defaultdict(int)
        
        for url, score in self.pagerank_scores.items():
            section = self.section_mapping.get(url, 'other')
            section_pr[section] += score
            section_counts[section] += 1
        
        total_pr = sum(section_pr.values())
        
        data = []
        for section, pr_score in sorted(section_pr.items(), key=lambda x: x[1], reverse=True):
            business_value = self.section_analyzer.get_business_value(section)
            conversion_score = self.section_analyzer.get_conversion_score(section)
            percentage = (pr_score / total_pr * 100) if total_pr > 0 else 0
            
            data.append({
                'section': section,
                'pagerank': pr_score,
                'percentage': percentage,
                'business_value': business_value,
                'conversion_score': conversion_score,
                'page_count': section_counts[section],
                'avg_pr': pr_score / section_counts[section] if section_counts[section] > 0 else 0
            })
        
        df = pd.DataFrame(data)
        
        # Create enhanced bar chart
        fig = px.bar(
            df,
            x='section',
            y='pagerank',
            color='business_value',
            color_discrete_map={
                'critical': '#dc2626',
                'high': '#ea580c',
                'medium': '#ca8a04',
                'low': '#16a34a',
                'minimal': '#9ca3af'
            },
            title='📊 PageRank Distribution by Section (with Business Value)',
            labels={'pagerank': 'PageRank Score', 'section': 'Section'},
            hover_data=['conversion_score', 'page_count', 'avg_pr']
        )
        
        # Add percentage labels
        fig.update_traces(
            texttemplate='%{customdata[0]:.1f}%',
            textposition='outside',
            customdata=df[['percentage']].values
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_tickangle=-45,
            legend_title="Business Value"
        )
        
        return fig
    
    def create_top_pages_chart(self):
        """Enhanced top pages visualization"""
        top_pages = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        
        data = []
        for i, (url, score) in enumerate(top_pages, 1):
            section = self.section_mapping.get(url, 'other')
            page_data = self.page_data.get(url, {})
            business_value = self.section_analyzer.get_business_value(section)
            conversion_score = self.section_analyzer.get_conversion_score(section)
            
            # Create readable page name
            parsed = urlparse(url)
            page_name = parsed.path.split('/')[-1] or 'Homepage'
            if not page_name or page_name in ['index.html', 'index.php']:
                page_name = 'Homepage'
            
            data.append({
                'rank': i,
                'page': page_name[:25] + '...' if len(page_name) > 25 else page_name,
                'pagerank': score,
                'section': section,
                'business_value': business_value,
                'conversion_score': conversion_score,
                'url': url,
                'word_count': page_data.get('word_count', 0),
                'internal_links': page_data.get('internal_links', 0)
            })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='page',
            y='pagerank',
            color='business_value',
            color_discrete_map={
                'critical': '#dc2626',
                'high': '#ea580c',
                'medium': '#ca8a04',
                'low': '#16a34a',
                'minimal': '#9ca3af'
            },
            title='🏆 Top 20 Pages by PageRank Score',
            labels={'pagerank': 'PageRank Score', 'page': 'Page'},
            hover_data=['conversion_score', 'section', 'word_count', 'internal_links']
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            showlegend=True
        )
        
        return fig
    
    def create_business_value_pie(self):
        """Enhanced business value distribution"""
        business_values = defaultdict(float)
        business_counts = defaultdict(int)
        
        for url, score in self.pagerank_scores.items():
            section = self.section_mapping.get(url, 'other')
            business_value = self.section_analyzer.get_business_value(section)
            business_values[business_value] += score
            business_counts[business_value] += 1
        
        # Create labels with counts and percentages
        labels = []
        values = []
        colors = {
            'critical': '#dc2626',
            'high': '#ea580c',
            'medium': '#ca8a04',
            'low': '#16a34a',
            'minimal': '#9ca3af'
        }
        
        total_pr = sum(business_values.values())
        for value_level in ['critical', 'high', 'medium', 'low', 'minimal']:
            if value_level in business_values:
                pr_amount = business_values[value_level]
                page_count = business_counts[value_level]
                percentage = (pr_amount / total_pr * 100) if total_pr > 0 else 0
                
                labels.append(f"{value_level.title()}<br>({page_count} pages)<br>{percentage:.1f}%")
                values.append(pr_amount)
        
        fig = px.pie(
            values=values,
            names=labels,
            title='💼 PageRank Distribution by Business Value',
            color_discrete_sequence=[colors[level] for level in ['critical', 'high', 'medium', 'low', 'minimal'] if level in business_values]
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='label',
            textfont_size=12
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_network_graph(self):
        """Enhanced network visualization with business value insights"""
        try:
            # Get top 30 pages for better network visualization
            top_pages = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:30]
            top_urls = [url for url, _ in top_pages]
            
            # Create subgraph
            subgraph = self.graph.subgraph(top_urls)
            
            if len(subgraph.nodes()) == 0:
                return go.Figure()
            
            # Enhanced layout with better positioning
            pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
            
            # Create edge traces with varying thickness
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in subgraph.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Calculate edge weight based on target PageRank
                    target_pr = self.pagerank_scores.get(edge[1], 0)
                    edge_weights.append(max(0.5, target_pr * 1000))
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color='rgba(100, 100, 100, 0.6)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            # Enhanced node traces with business value coloring
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            node_hover = []
            
            color_map = {
                'critical': '#dc2626',
                'high': '#ea580c',
                'medium': '#ca8a04',
                'low': '#16a34a',
                'minimal': '#9ca3af'
            }
            
            for node in subgraph.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Get node properties
                    pr_score = self.pagerank_scores.get(node, 0)
                    section = self.section_mapping.get(node, 'other')
                    business_value = self.section_analyzer.get_business_value(section)
                    conversion_score = self.section_analyzer.get_conversion_score(section)
                    
                    # Size based on PageRank (enhanced scaling)
                    node_size.append(max(15, pr_score * 2000))
                    
                    # Color based on business value
                    node_color.append(color_map.get(business_value, '#6b7280'))
                    
                    # Create readable node label
                    parsed = urlparse(node)
                    page_name = parsed.path.split('/')[-1] or 'Home'
                    if len(page_name) > 20:
                        page_name = page_name[:17] + '...'
                    
                    node_text.append(page_name)
                    
                    # Enhanced hover info
                    page_data = self.page_data.get(node, {})
                    title = page_data.get('title', 'No Title')
                    
                    node_hover.append(
                        f"<b>{page_name}</b><br>" +
                        f"Title: {title[:50]}{'...' if len(title) > 50 else ''}<br>" +
                        f"PageRank: {pr_score:.6f}<br>" +
                        f"Section: {section}<br>" +
                        f"Business Value: {business_value.title()}<br>" +
                        f"Conversion Score: {conversion_score}/100<br>" +
                        f"Word Count: {page_data.get('word_count', 0):,}<br>" +
                        f"Internal Links: {page_data.get('internal_links', 0)}"
                    )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_hover,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color="white", family="Inter"),
                marker=dict(
                    size=node_size,
                    color=node_color,
                    opacity=0.9,
                    line=dict(width=2, color="white")
                ),
                showlegend=False
            )
            
            # Create enhanced figure
            fig = go.Figure(data=[edge_trace, node_trace])
            
            fig.update_layout(
                title={
                    'text': '🕸️ PageRank Flow Network - Top 30 Pages',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'family': 'Inter'}
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=30,l=10,r=10,t=50),
                annotations=[dict(
                    text="Node size = PageRank score | Color = Business value (Red=Critical, Orange=High, Yellow=Medium, Green=Low, Gray=Minimal)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.05,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=11, family="Inter")
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=700,
                plot_bgcolor='rgba(250,250,250,0.95)'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating network graph: {str(e)}")
            return go.Figure()
    
    def create_section_matrix_heatmap(self):
        """Enhanced section linking heatmap"""
        section_links = defaultdict(lambda: defaultdict(int))
        
        for source, target in self.graph.edges():
            source_section = self.section_mapping.get(source, 'other')
            target_section = self.section_mapping.get(target, 'other')
            section_links[source_section][target_section] += 1
        
        sections = sorted(set(self.section_mapping.values()))
        
        # Create matrix with percentage calculation
        matrix = []
        row_totals = []
        for source in sections:
            row = []
            row_total = sum(section_links[source].values())
            row_totals.append(row_total)
            
            for target in sections:
                count = section_links[source][target]
                percentage = (count / row_total * 100) if row_total > 0 else 0
                row.append(percentage)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=sections,
            y=sections,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Link %"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔥 Section-to-Section Linking Heatmap (% Distribution)',
            xaxis_title='Target Section',
            yaxis_title='Source Section',
            height=600,
            font=dict(family="Inter")
        )
        
        return fig

def handle_csv_upload_with_mapping():
    """Enhanced CSV upload with flexible column mapping and data validation"""
    st.subheader("📁 Priority Pages & Data Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload your priority pages, Ahrefs data, GSC data, or any relevant CSV"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV with error handling
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            if df.empty:
                st.error("❌ The uploaded file is empty")
                return None
            
            st.success(f"✅ File uploaded successfully! {len(df)} rows, {len(df.columns)} columns")
            
            # Enhanced preview with data types
            st.markdown("**📊 File Preview & Analysis:**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.markdown("**📈 Data Summary:**")
                st.markdown(f"- **Rows:** {len(df):,}")
                st.markdown(f"- **Columns:** {len(df.columns)}")
                st.markdown(f"- **Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                
                # Show data types
                st.markdown("**🔍 Column Types:**")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    st.markdown(f"- `{col}`: {dtype} ({null_count} nulls)")
            
            # Enhanced column mapping interface
            st.markdown("---")
            st.markdown("**🔧 Smart Column Mapping:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Auto-detect likely columns
            url_candidates = [col for col in df.columns if any(term in col.lower() for term in ['url', 'link', 'page', 'address'])]
            keyword_candidates = [col for col in df.columns if any(term in col.lower() for term in ['keyword', 'query', 'term', 'phrase'])]
            metric_candidates = [col for col in df.columns if any(term in col.lower() for term in ['click', 'impression', 'traffic', 'rank', 'position', 'volume', 'score'])]
            
            with col1:
                url_column = st.selectbox(
                    "📍 URL Column",
                    options=['None'] + df.columns.tolist(),
                    index=1 if url_candidates and url_candidates[0] in df.columns else 0,
                    help="Select the column containing URLs or page paths"
                )
            
            with col2:
                keyword_column = st.selectbox(
                    "🔑 Keywords Column",
                    options=['None'] + df.columns.tolist(),
                    index=df.columns.tolist().index(keyword_candidates[0]) + 1 if keyword_candidates and keyword_candidates[0] in df.columns else 0,
                    help="Select the column with target keywords or search queries"
                )
            
            with col3:
                metric_column = st.selectbox(
                    "📊 Primary Metric Column",
                    options=['None'] + df.columns.tolist(),
                    index=df.columns.tolist().index(metric_candidates[0]) + 1 if metric_candidates and metric_candidates[0] in df.columns else 0,
                    help="Select main performance metric (clicks, traffic, rankings, etc.)"
                )
            
            with col4:
                secondary_metric_column = st.selectbox(
                    "📈 Secondary Metric (Optional)",
                    options=['None'] + df.columns.tolist(),
                    help="Select additional metric for analysis"
                )
            
            # Advanced processing options
            st.markdown("---")
            st.markdown("**⚙️ Advanced Processing Options:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                remove_duplicates = st.checkbox("🔄 Remove duplicate URLs", value=True)
                clean_urls = st.checkbox("🧹 Clean and normalize URLs", value=True)
                filter_nulls = st.checkbox("🚫 Remove rows with null URLs", value=True)
            
            with col2:
                limit_rows = st.checkbox("⚡ Limit rows for performance")
                if limit_rows:
                    max_rows = st.slider("Max rows to process", 100, 10000, 2000, step=100)
                else:
                    max_rows = len(df)
                
                min_metric_threshold = st.checkbox("📊 Apply minimum metric threshold")
                if min_metric_threshold and metric_column != 'None':
                    try:
                        metric_min = st.number_input(
                            f"Minimum {metric_column} value",
                            min_value=0.0,
                            value=0.0,
                            step=1.0
                        )
                    except:
                        metric_min = 0
                else:
                    metric_min = 0
            
            with col3:
                url_filter = st.text_input(
                    "🔍 URL Contains Filter (optional)",
                    placeholder="e.g., /blog/, product",
                    help="Only include URLs containing this text"
                )
                
                exclude_filter = st.text_input(
                    "🚫 Exclude URLs Containing",
                    placeholder="e.g., admin, test, staging",
                    help="Exclude URLs containing this text"
                )
            
            # Process the data if URL column is selected
            if url_column != 'None':
                try:
                    processed_df = df.copy()
                    
                    # Apply filters first
                    if filter_nulls:
                        processed_df = processed_df.dropna(subset=[url_column])
                    
                    if url_filter:
                        processed_df = processed_df[processed_df[url_column].str.contains(url_filter, case=False, na=False)]
                    
                    if exclude_filter:
                        processed_df = processed_df[~processed_df[url_column].str.contains(exclude_filter, case=False, na=False)]
                    
                    # Apply metric threshold
                    if min_metric_threshold and metric_column != 'None':
                        try:
                            processed_df[metric_column] = pd.to_numeric(processed_df[metric_column], errors='coerce')
                            processed_df = processed_df[processed_df[metric_column] >= metric_min]
                        except:
                            st.warning("⚠️ Could not apply metric threshold - column contains non-numeric data")
                    
                    # Limit rows
                    if limit_rows and len(processed_df) > max_rows:
                        processed_df = processed_df.head(max_rows)
                        st.info(f"ℹ️ Limited to {max_rows} rows for performance")
                    
                    # Create standardized result dataframe
                    result_df = pd.DataFrame()
                    result_df['URL'] = processed_df[url_column].astype(str)
                    
                    # Clean URLs if requested
                    if clean_urls:
                        result_df['URL'] = result_df['URL'].str.strip()
                        # Handle relative URLs
                        mask = ~result_df['URL'].str.startswith(('http://', 'https://'))
                        result_df.loc[mask, 'URL'] = 'https://' + result_df.loc[mask, 'URL'].str.lstrip('/')
                    
                    # Add other columns
                    if keyword_column != 'None':
                        result_df['Target Keywords'] = processed_df[keyword_column].fillna('').astype(str)
                    else:
                        result_df['Target Keywords'] = ''
                    
                    if metric_column != 'None':
                        try:
                            result_df['Primary Metric'] = pd.to_numeric(processed_df[metric_column], errors='coerce').fillna(0)
                        except:
                            result_df['Primary Metric'] = processed_df[metric_column].astype(str)
                    else:
                        result_df['Primary Metric'] = 0
                    
                    if secondary_metric_column != 'None':
                        try:
                            result_df['Secondary Metric'] = pd.to_numeric(processed_df[secondary_metric_column], errors='coerce').fillna(0)
                        except:
                            result_df['Secondary Metric'] = processed_df[secondary_metric_column].astype(str)
                    else:
                        result_df['Secondary Metric'] = 0
                    
                    # Remove duplicates if requested
                    if remove_duplicates:
                        initial_count = len(result_df)
                        result_df = result_df.drop_duplicates(subset=['URL'])
                        final_count = len(result_df)
                        if initial_count != final_count:
                            st.info(f"🔄 Removed {initial_count - final_count} duplicate URLs")
                    
                    # Final data validation
                    result_df = result_df[result_df['URL'].str.len() > 0]  # Remove empty URLs
                    
                    if len(result_df) == 0:
                        st.error("❌ No valid data remaining after processing")
                        return None
                    
                    # Show processed data preview
                    st.markdown("---")
                    st.markdown("**✅ Processed Data Preview:**")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.dataframe(result_df.head(10), use_container_width=True)
                    
                    with col2:
                        st.markdown("**📊 Final Stats:**")
                        st.markdown(f"- **Records:** {len(result_df):,}")
                        st.markdown(f"- **Unique URLs:** {result_df['URL'].nunique():,}")
                        if keyword_column != 'None':
                            st.markdown(f"- **With Keywords:** {(result_df['Target Keywords'] != '').sum():,}")
                        if metric_column != 'None':
                            try:
                                avg_metric = result_df['Primary Metric'].mean()
                                st.markdown(f"- **Avg {metric_column}:** {avg_metric:.1f}")
                            except:
                                pass
                    
                    st.success(f"✅ Successfully processed {len(result_df):,} records")
                    
                    return result_df
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    return None
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None
    
    return None

def generate_enhanced_ai_recommendations(pagerank_scores, section_stats, priority_analysis, page_data, graph, openai_key, website_url):
    """Generate comprehensive AI recommendations with advanced analysis"""
    if not openai_key:
        return "OpenAI API key not provided for enhanced recommendations."
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # Comprehensive data preparation
        total_pages = len(pagerank_scores)
        total_links = len(graph.edges())
        
        # Calculate advanced metrics
        total_pr = sum(s['total_pr'] for s in section_stats) if section_stats else 1
        critical_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'critical') if section_stats else 0
        high_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'high') if section_stats else 0
        low_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] in ['low', 'minimal']) if section_stats else 0
        
        efficiency_score = ((critical_pr + high_pr) / total_pr * 100) if total_pr > 0 else 0
        waste_percentage = (low_pr / total_pr * 100) if total_pr > 0 else 0
        
        # Analyze top performing pages
        top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        top_sections_analysis = {}
        for url, score in top_pages:
            section = next((s['section'] for s in section_stats if s['section'] in url.lower()), 'unknown')
            if section not in top_sections_analysis:
                top_sections_analysis[section] = {'count': 0, 'total_pr': 0}
            top_sections_analysis[section]['count'] += 1
            top_sections_analysis[section]['total_pr'] += score
        
        # Content analysis
        avg_word_count = np.mean([data.get('word_count', 0) for data in page_data.values()]) if page_data else 0
        avg_internal_links = np.mean([data.get('internal_links', 0) for data in page_data.values()]) if page_data else 0
        
        # Network analysis
        if total_pages > 1:
            density = nx.density(graph) * 100
            try:
                avg_clustering = nx.average_clustering(graph.to_undirected()) * 100
            except:
                avg_clustering = 0
        else:
            density = 0
            avg_clustering = 0
        
        # Priority pages analysis
        priority_insights = ""
        if priority_analysis:
            found_count = len([p for p in priority_analysis if p['found']])
            total_priority = len(priority_analysis)
            avg_priority_pr = np.mean([p['pagerank'] for p in priority_analysis if p['found']]) if found_count > 0 else 0
            
            priority_insights = f"""
        **Priority Pages Performance:**
        - Found {found_count} of {total_priority} priority pages ({found_count/total_priority*100:.1f}% coverage)
        - Average PageRank of found priority pages: {avg_priority_pr:.6f}
        - Missing priority pages may indicate crawl limitations or broken internal linking
        """
        
        # Create comprehensive analysis prompt
        enhanced_prompt = f"""
        As a senior technical SEO consultant and PageRank optimization specialist, provide a comprehensive, data-driven analysis and actionable recommendations for this website: {website_url}

        **COMPREHENSIVE WEBSITE ANALYSIS:**
        
        **Core Metrics:**
        - Website: {website_url}
        - Total pages analyzed: {total_pages:,}
        - Total internal links: {total_links:,}
        - Average word count per page: {avg_word_count:.0f}
        - Average internal links per page: {avg_internal_links:.1f}
        - Network density: {density:.2f}%
        - Network clustering coefficient: {avg_clustering:.2f}%
        
        **PageRank Distribution Analysis:**
        - Overall PageRank efficiency: {efficiency_score:.1f}%
        - Critical/High-value page concentration: {(critical_pr + high_pr):.4f} ({(critical_pr + high_pr)/total_pr*100:.1f}%)
        - PageRank waste in low-value sections: {waste_percentage:.1f}%
        - Sections identified: {len(section_stats) if section_stats else 0}
        
        **Top Performing Sections:**
        {chr(10).join([f"- {s['section']}: {s['total_pr']:.4f} PageRank ({s['percentage']:.1f}%), {s['page_count']} pages, Business Value: {s['business_value']}" for s in (section_stats[:8] if section_stats else [])])}
        
        {priority_insights}
        
        **STRATEGIC ANALYSIS REQUIRED:**
        
        1. **IMMEDIATE CRITICAL ACTIONS (Week 1-2):**
           - Identify the top 3 most critical PageRank distribution issues
           - Provide specific technical fixes with expected impact percentages
           - Address any urgent PageRank waste problems (>20% waste is critical)
           
        2. **HIGH-IMPACT OPTIMIZATIONS (Month 1-2):**
           - Strategic internal linking improvements
           - Content hub development recommendations
           - Navigation and site architecture enhancements
           - Specific page-to-page linking strategies
           
        3. **LONG-TERM STRATEGIC IMPROVEMENTS (Month 3-6):**
           - Content strategy alignment with PageRank flow
           - Technical infrastructure optimization
           - Advanced internal linking schemes
           - Conversion funnel optimization through PageRank distribution
           
        4. **TECHNICAL IMPLEMENTATION GUIDE:**
           - Provide specific HTML/technical changes needed
           - Prioritized link building internal roadmap
           - Content consolidation/expansion recommendations
           - Site architecture improvements
           
        5. **EXPECTED ROI & IMPACT PROJECTIONS:**
           - Quantified improvement estimates
           - Traffic impact predictions
           - Conversion rate optimization potential
           - Timeline for seeing results
           
        6. **MONITORING & MEASUREMENT:**
           - Key metrics to track post-implementation
           - Tools and methods for ongoing optimization
           - Success indicators and benchmarks
           
        **CRITICAL REQUIREMENTS:**
        - All recommendations must be specific and actionable
        - Include expected impact percentages where possible
        - Prioritize recommendations by ROI potential
        - Consider technical difficulty vs. impact ratio
        - Address both immediate wins and long-term strategic gains
        - Focus on measurable outcomes
        
        Provide detailed, professional analysis with specific implementation steps for each recommendation.
        """
        
        # Generate enhanced recommendations
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a world-class technical SEO consultant specializing in PageRank optimization, internal linking strategies, and website architecture. You have 15+ years of experience and have successfully optimized hundreds of enterprise websites. Provide detailed, actionable, and measurable recommendations with specific implementation guidance."
                },
                {
                    "role": "user", 
                    "content": enhanced_prompt
                }
            ],
            max_tokens=2500,
            temperature=0.3  # Lower temperature for more focused, technical responses
        )
        
        ai_recommendations = response.choices[0].message.content
        
        # Add technical appendix
        technical_appendix = f"""
        
        ---
        
        **TECHNICAL DATA APPENDIX:**
        
        **Site Architecture Metrics:**
        - Average page depth: {np.mean([data.get('route_depth', 0) for data in page_data.values()]) if page_data else 0:.1f}
        - Link distribution variance: {np.var([data.get('internal_links', 0) for data in page_data.values()]) if page_data else 0:.1f}
        - Content length variance: {np.var([data.get('word_count', 0) for data in page_data.values()]) if page_data else 0:.0f}
        
        **Network Analysis:**
        - Network connectivity: {density:.2f}% (Target: >15% for good internal linking)
        - Clustering: {avg_clustering:.2f}% (Higher values indicate topic-focused linking)
        - Link equity distribution: {'Well distributed' if efficiency_score > 60 else 'Concentrated' if efficiency_score > 40 else 'Poorly distributed'}
        
        **Optimization Potential Score: {min(100, max(0, 100 - efficiency_score + (waste_percentage/2))):.0f}/100**
        """
        
        return ai_recommendations + technical_appendix
        
    except Exception as e:
        error_details = str(e)
        return f"""
        **AI Recommendation Generation Error**
        
        Unfortunately, there was an issue generating AI recommendations: {error_details}
        
        **Manual Analysis Summary:**
        - Total pages analyzed: {total_pages:,}
        - PageRank efficiency: {efficiency_score:.1f}%
        - Waste percentage: {waste_percentage:.1f}%
        
        **Basic Recommendations:**
        1. Focus on reducing PageRank waste if >15%
        2. Improve internal linking to high-value pages
        3. Consolidate thin content pages
        4. Optimize navigation structure
        5. Monitor and iterate on changes
        
        Please check your OpenAI API key and try again for detailed recommendations.
        """

def generate_comprehensive_csv_reports(crawler, pagerank_scores, section_mapping, section_analyzer, priority_df=None):
    """Generate multiple CSV reports for comprehensive analysis"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reports = {}
    
    try:
        # 1. Executive Summary Report
        summary_data = {
            'Metric': [
                'Analysis Date',
                'Website URL',
                'Total Pages Analyzed',
                'Total Internal Links',
                'Unique Sections Found',
                'Crawl Errors',
                'Analysis Duration (seconds)',
                'Average PageRank Score',
                'PageRank Efficiency Score',
                'PageRank Waste Percentage',
                'Network Density',
                'Average Words per Page',
                'Average Internal Links per Page'
            ],
            'Value': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                crawler.seed_url,
                len(pagerank_scores),
                len(crawler.graph.edges()),
                len(set(section_mapping.values())),
                crawler.crawl_stats.get('errors', 0),
                (crawler.crawl_stats['end_time'] - crawler.crawl_stats['start_time']).total_seconds() if crawler.crawl_stats.get('end_time') and crawler.crawl_stats.get('start_time') else 0,
                np.mean(list(pagerank_scores.values())) if pagerank_scores else 0,
                0,  # Will be calculated
                0,  # Will be calculated
                nx.density(crawler.graph) * 100 if len(crawler.graph.nodes()) > 1 else 0,
                np.mean([data.get('word_count', 0) for data in crawler.page_data.values()]) if crawler.page_data else 0,
                np.mean([data.get('internal_links', 0) for data in crawler.page_data.values()]) if crawler.page_data else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        reports['executive_summary'] = summary_df
        
        # 2. Detailed Pages Analysis Report
        pages_data = []
        for i, (url, score) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True), 1):
            page_info = crawler.page_data.get(url, {})
            section = section_mapping.get(url, 'other')
            business_value = section_analyzer.get_business_value(section)
            conversion_score = section_analyzer.get_conversion_score(section)
            
            pages_data.append({
                'Rank': i,
                'URL': url,
                'PageRank_Score': f"{score:.8f}",
                'Section': section,
                'Business_Value': business_value,
                'Conversion_Score': conversion_score,
                'Page_Title': page_info.get('title', ''),
                'Meta_Description': page_info.get('meta_description', ''),
                'H1_Text': page_info.get('h1', ''),
                'Word_Count': page_info.get('word_count', 0),
                'Route_Depth': page_info.get('route_depth', 0),
                'Internal_Links_Count': page_info.get('internal_links', 0),
                'External_Links_Count': page_info.get('external_links', 0),
                'Status_Code': page_info.get('status_code', 200),
                'Content_Length': page_info.get('content_length', 0),
                'Images_Count': page_info.get('images', 0),
                'Headers_Count': page_info.get('headers', 0)
            })
        
        pages_df = pd.DataFrame(pages_data)
        reports['detailed_pages'] = pages_df
        
        # 3. Section Performance Analysis
        section_data = []
        total_pr = sum(pagerank_scores.values())
        
        section_pr = defaultdict(float)
        section_counts = defaultdict(int)
        section_metrics = defaultdict(lambda: {'word_count': [], 'internal_links': [], 'conversion_scores': []})
        
        for url, score in pagerank_scores.items():
            section = section_mapping.get(url, 'other')
            section_pr[section] += score
            section_counts[section] += 1
            
            page_info = crawler.page_data.get(url, {})
            section_metrics[section]['word_count'].append(page_info.get('word_count', 0))
            section_metrics[section]['internal_links'].append(page_info.get('internal_links', 0))
            section_metrics[section]['conversion_scores'].append(section_analyzer.get_conversion_score(section))
        
        for section in sorted(section_pr.keys(), key=lambda x: section_pr[x], reverse=True):
            business_value = section_analyzer.get_business_value(section)
            avg_conversion = np.mean(section_metrics[section]['conversion_scores']) if section_metrics[section]['conversion_scores'] else 0
            
            section_data.append({
                'Section': section,
                'Total_PageRank': f"{section_pr[section]:.8f}",
                'Page_Count': section_counts[section],
                'Average_PageRank': f"{section_pr[section] / section_counts[section]:.8f}" if section_counts[section] > 0 else "0",
                'Percentage_of_Total_PR': f"{(section_pr[section] / total_pr * 100):.2f}%" if total_pr > 0 else "0%",
                'Business_Value': business_value,
                'Average_Conversion_Score': f"{avg_conversion:.1f}",
                'Average_Word_Count': f"{np.mean(section_metrics[section]['word_count']):.0f}" if section_metrics[section]['word_count'] else "0",
                'Average_Internal_Links': f"{np.mean(section_metrics[section]['internal_links']):.1f}" if section_metrics[section]['internal_links'] else "0",
                'Optimization_Priority': 'High' if business_value in ['critical', 'high'] and section_pr[section] / total_pr < 0.1 else 'Medium' if business_value == 'medium' else 'Low'
            })
        
        sections_df = pd.DataFrame(section_data)
        reports['section_analysis'] = sections_df
        
        # 4. Priority Pages Analysis (if provided)
        if priority_df is not None and not priority_df.empty:
            priority_analysis_data = []
            
            for _, row in priority_df.iterrows():
                url = row.get('URL', '')
                keywords = str(row.get('Target Keywords', ''))
                primary_metric = row.get('Primary Metric', 0)
                
                if url in pagerank_scores:
                    score = pagerank_scores[url]
                    rank = sorted(pagerank_scores.values(), reverse=True).index(score) + 1
                    section = section_mapping.get(url, 'other')
                    business_value = section_analyzer.get_business_value(section)
                    page_info = crawler.page_data.get(url, {})
                    
                    priority_analysis_data.append({
                        'Priority_URL': url,
                        'Found_in_Crawl': 'Yes',
                        'PageRank_Score': f"{score:.8f}",
                        'PageRank_Rank': rank,
                        'Percentile': f"{(rank / len(pagerank_scores) * 100):.1f}%",
                        'Section': section,
                        'Business_Value': business_value,
                        'Target_Keywords': keywords,
                        'Primary_Metric_Value': primary_metric,
                        'Page_Title': page_info.get('title', ''),
                        'Word_Count': page_info.get('word_count', 0),
                        'Internal_Links': page_info.get('internal_links', 0),
                        'Route_Depth': page_info.get('route_depth', 0),
                        'Optimization_Status': 'Good' if rank <= len(pagerank_scores) * 0.2 else 'Needs Improvement' if rank <= len(pagerank_scores) * 0.5 else 'Poor'
                    })
                else:
                    priority_analysis_data.append({
                        'Priority_URL': url,
                        'Found_in_Crawl': 'No',
                        'PageRank_Score': '0',
                        'PageRank_Rank': 'Not Found',
                        'Percentile': 'N/A',
                        'Section': 'Not Found',
                        'Business_Value': 'Unknown',
                        'Target_Keywords': keywords,
                        'Primary_Metric_Value': primary_metric,
                        'Page_Title': 'Not Found',
                        'Word_Count': 0,
                        'Internal_Links': 0,
                        'Route_Depth': 0,
                        'Optimization_Status': 'Missing - Needs Internal Links'
                    })
            
            priority_analysis_df = pd.DataFrame(priority_analysis_data)
            reports['priority_pages'] = priority_analysis_df
        
        # 5. Internal Linking Opportunities
        linking_opportunities = []
        
        # Find high-PR pages that could link to priority pages
        if priority_df is not None and not priority_df.empty:
            high_pr_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:50]
            priority_urls = set(priority_df['URL'].tolist())
            
            for high_pr_url, high_pr_score in high_pr_pages:
                if high_pr_url not in priority_urls:
                    high_pr_section = section_mapping.get(high_pr_url, 'other')
                    
                    # Find priority pages that could benefit from this high-PR page's links
                    for priority_url in priority_urls:
                        if priority_url in pagerank_scores:
                            priority_section = section_mapping.get(priority_url, 'other')
                            priority_score = pagerank_scores[priority_url]
                            
                            # Calculate potential impact
                            potential_impact = high_pr_score * 0.15 * 0.85  # Simplified PageRank transfer calculation
                            
                            linking_opportunities.append({
                                'Source_URL': high_pr_url,
                                'Source_PageRank': f"{high_pr_score:.8f}",
                                'Source_Section': high_pr_section,
                                'Target_URL': priority_url,
                                'Target_PageRank': f"{priority_score:.8f}",
                                'Target_Section': priority_section,
                                'Potential_PageRank_Boost': f"{potential_impact:.8f}",
                                'Impact_Percentage': f"{(potential_impact / priority_score * 100):.1f}%" if priority_score > 0 else "N/A",
                                'Recommendation': f"Add contextual link from {high_pr_section} to {priority_section}",
                                'Priority': 'High' if potential_impact > priority_score * 0.1 else 'Medium' if potential_impact > priority_score * 0.05 else 'Low'
                            })
        
        if linking_opportunities:
            # Sort by potential impact and take top 100
            linking_opportunities.sort(key=lambda x: float(x['Potential_PageRank_Boost']), reverse=True)
            linking_df = pd.DataFrame(linking_opportunities[:100])
            reports['linking_opportunities'] = linking_df
        
        # 6. Technical SEO Issues Report
        technical_issues = []
        
        for url, page_info in crawler.page_data.items():
            issues = []
            
            # Check for common technical issues
            if not page_info.get('title'):
                issues.append('Missing Title Tag')
            elif len(page_info.get('title', '')) < 30:
                issues.append('Title Too Short (<30 chars)')
            elif len(page_info.get('title', '')) > 60:
                issues.append('Title Too Long (>60 chars)')
            
            if not page_info.get('meta_description'):
                issues.append('Missing Meta Description')
            elif len(page_info.get('meta_description', '')) < 120:
                issues.append('Meta Description Too Short')
            elif len(page_info.get('meta_description', '')) > 160:
                issues.append('Meta Description Too Long')
            
            if not page_info.get('h1'):
                issues.append('Missing H1 Tag')
            
            if page_info.get('word_count', 0) < 200:
                issues.append('Thin Content (<200 words)')
            
            if page_info.get('internal_links', 0) < 3:
                issues.append('Low Internal Link Count (<3)')
            
            if page_info.get('route_depth', 0) > 4:
                issues.append('Deep URL Structure (>4 levels)')
            
            if issues:
                section = section_mapping.get(url, 'other')
                pr_score = pagerank_scores.get(url, 0)
                
                technical_issues.append({
                    'URL': url,
                    'PageRank_Score': f"{pr_score:.8f}",
                    'Section': section,
                    'Issues': '; '.join(issues),
                    'Issue_Count': len(issues),
                    'Priority': 'High' if pr_score > np.percentile(list(pagerank_scores.values()), 80) else 'Medium' if pr_score > np.percentile(list(pagerank_scores.values()), 50) else 'Low'
                })
        
        if technical_issues:
            technical_df = pd.DataFrame(technical_issues)
            reports['technical_issues'] = technical_df
        
        return reports, timestamp
        
    except Exception as e:
        st.error(f"Error generating CSV reports: {str(e)}")
        return {}, timestamp

def main():
    """Enhanced main application with comprehensive features"""
    
    # Stunning header with animation
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced PageRank SEO Analyzer</h1>
        <h2>Ultimate Edition with Enhanced AI Recommendations</h2>
        <p>Complete analysis of the 5 critical PageRank distribution questions</p>
        <p><strong>✨ Now with comprehensive CSV reports and advanced AI insights ✨</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar configuration
    st.sidebar.markdown("### 🔧 Analysis Configuration")
    
    # OpenAI API Key with enhanced description
    openai_key = st.sidebar.text_input(
        "🤖 OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for advanced AI recommendations and insights"
    )
    
    if openai_key:
        st.sidebar.success("✅ Enhanced AI recommendations enabled")
        st.sidebar.markdown("💡 *You'll get comprehensive strategic analysis and implementation guidance*")
    else:
        st.sidebar.info("💡 Add OpenAI key for advanced AI recommendations")
    
    # Website URL
    website_url = st.sidebar.text_input(
        "🌐 Website URL",
        placeholder="https://example.com",
        help="Enter the root URL of the website you want to analyze"
    )
    
    # Enhanced crawling settings
    st.sidebar.markdown("### ⚙️ Crawling Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_pages = st.slider("📄 Max Pages", 100, 2000, 500, step=50)
        max_depth = st.slider("🕳️ Max Depth", 1, 5, 3)
    
    with col2:
        crawl_delay = st.slider("⏱️ Delay (sec)", 0.1, 2.0, 0.2, step=0.1)
        respect_robots = st.checkbox("🤖 Respect robots.txt", value=True)
    
    # Advanced options
    with st.sidebar.expander("🔬 Advanced Options"):
        follow_redirects = st.checkbox("↩️ Follow redirects", value=True)
        analyze_images = st.checkbox("🖼️ Analyze images", value=False)
        extract_schema = st.checkbox("📋 Extract schema markup", value=False)
        deep_content_analysis = st.checkbox("📖 Deep content analysis", value=False)
    
    # CSV Upload with enhanced features
    priority_df = handle_csv_upload_with_mapping()
    
    # Main analysis with enhanced progress tracking
    if st.sidebar.button("🚀 Start Comprehensive Analysis", type="primary"):
        if not website_url:
            st.error("❌ Please enter a website URL to analyze")
            return
        
        # Validate URL
        try:
            parsed_url = urlparse(website_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                st.error("❌ Please enter a valid URL (e.g., https://example.com)")
                return
        except:
            st.error("❌ Invalid URL format")
            return
        
        # Initialize enhanced components
        crawler = OptimizedCrawler(website_url, max_pages, max_depth, crawl_delay)
        section_analyzer = AdvancedSectionAnalyzer()
        
        # Step 1: Enhanced crawling
        st.markdown("## 🕷️ Advanced Website Crawling & Data Collection")
        
        start_time = time.time()
        crawled_urls = crawler.crawl()
        crawl_duration = time.time() - start_time
        
        if not crawled_urls:
            st.error("❌ No pages were successfully crawled. Please check the URL and try again.")
            return
        
        # Enhanced crawl summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 Pages Crawled</h3>
                <h2>{len(crawled_urls):,}</h2>
                <p>{len(crawled_urls)/max_pages*100:.1f}% of target</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 Links Found</h3>
                <h2>{crawler.crawl_stats['links_found']:,}</h2>
                <p>{crawler.crawl_stats['links_found']/len(crawled_urls):.1f} per page</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Crawl Speed</h3>
                <h2>{len(crawled_urls)/crawl_duration:.1f}</h2>
                <p>pages per second</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            error_rate = (crawler.crawl_stats['errors'] / (len(crawled_urls) + crawler.crawl_stats['errors']) * 100) if len(crawled_urls) > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Success Rate</h3>
                <h2>{100-error_rate:.1f}%</h2>
                <p>{crawler.crawl_stats['errors']} errors</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Step 2: Enhanced section detection
        st.markdown("## 🔍 Intelligent Section Detection & Categorization")
        
        section_mapping = {}
        
        with st.spinner("🧠 Analyzing website structure with advanced algorithms..."):
            for url in crawled_urls:
                page_data = crawler.page_data.get(url, {})
                title = page_data.get('title', '')
                h1 = page_data.get('h1', '')
                meta_desc = page_data.get('meta_description', '')
                
                section = section_analyzer.categorize_url(url, title, meta_desc, h1)
                section_mapping[url] = section
        
        sections_found = len(set(section_mapping.values()))
        
        # Section detection summary
        section_counts = Counter(section_mapping.values())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ Identified {sections_found} distinct sections across {len(crawled_urls)} pages")
            
            # Show section distribution
            section_dist_data = []
            for section, count in section_counts.most_common():
                business_value = section_analyzer.get_business_value(section)
                conversion_score = section_analyzer.get_conversion_score(section)
                
                section_dist_data.append({
                    'Section': section.title(),
                    'Pages': count,
                    'Percentage': f"{count/len(crawled_urls)*100:.1f}%",
                    'Business Value': business_value.title(),
                    'Conversion Score': f"{conversion_score}/100"
                })
            
            st.dataframe(pd.DataFrame(section_dist_data), use_container_width=True)
        
        with col2:
            # Business value pie chart preview
            business_value_counts = defaultdict(int)
            for section in section_mapping.values():
                business_value = section_analyzer.get_business_value(section)
                business_value_counts[business_value] += 1
            
            fig_bv_preview = px.pie(
                values=list(business_value_counts.values()),
                names=[name.title() for name in business_value_counts.keys()],
                title="Business Value Distribution",
                color_discrete_map={
                    'Critical': '#dc2626',
                    'High': '#ea580c',
                    'Medium': '#ca8a04',
                    'Low': '#16a34a',
                    'Minimal': '#9ca3af'
                }
            )
            fig_bv_preview.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bv_preview, use_container_width=True)
        
        # Step 3: Enhanced PageRank calculation
        st.markdown("## 📊 Advanced PageRank Computation")
        
        with st.spinner("🧮 Computing PageRank scores with advanced algorithms..."):
            if len(crawler.graph.edges()) == 0:
                st.warning("⚠️ No internal links found - using equal distribution")
                num_pages = len(crawled_urls)
                pagerank_scores = {url: 1.0/num_pages for url in crawled_urls}
            else:
                # Enhanced PageRank calculation with multiple damping factors
                pagerank_scores = nx.pagerank(crawler.graph, alpha=0.85, max_iter=100, tol=1e-6)
                
                # Normalize scores
                total_score = sum(pagerank_scores.values())
                if total_score > 0:
                    pagerank_scores = {url: score/total_score for url, score in pagerank_scores.items()}
        
        # PageRank calculation summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 PageRank Calculated</h3>
                <h2>{len(pagerank_scores):,}</h2>
                <p>pages scored</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_pr = np.mean(list(pagerank_scores.values())) if pagerank_scores else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Average Score</h3>
                <h2>{avg_pr:.6f}</h2>
                <p>mean PageRank</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_pr = max(pagerank_scores.values()) if pagerank_scores else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Highest Score</h3>
                                <h2>{max_pr:.6f}</h2>
                <p>top page score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.success(f"✅ PageRank calculated successfully for {len(pagerank_scores)} pages")
        
        # Step 4: Comprehensive analysis
        st.markdown("## 📈 Comprehensive Analysis & Insights")
        
        # Calculate enhanced section statistics
        section_stats = []
        total_pr = sum(pagerank_scores.values())
        section_pr = defaultdict(float)
        section_count = defaultdict(int)
        
        for url, section in section_mapping.items():
            pr_score = pagerank_scores.get(url, 0)
            section_pr[section] += pr_score
            section_count[section] += 1
        
        for section, pr_score in section_pr.items():
            business_value = section_analyzer.get_business_value(section)
            conversion_score = section_analyzer.get_conversion_score(section)
            percentage = (pr_score / total_pr * 100) if total_pr > 0 else 0
            
            section_stats.append({
                'section': section,
                'total_pr': pr_score,
                'page_count': section_count[section],
                'percentage': percentage,
                'business_value': business_value,
                'conversion_score': conversion_score
            })
        
        section_stats.sort(key=lambda x: x['total_pr'], reverse=True)
        
        # Enhanced priority analysis
        priority_analysis = None
        if priority_df is not None and not priority_df.empty:
            priority_analysis = []
            for _, row in priority_df.iterrows():
                url = row.get('URL', '')
                keywords = row.get('Target Keywords', '')
                primary_metric = row.get('Primary Metric', 0)
                
                if url in pagerank_scores:
                    score = pagerank_scores[url]
                    rank = sorted(pagerank_scores.values(), reverse=True).index(score) + 1
                    section = section_mapping.get(url, 'other')
                    priority_analysis.append({
                        'url': url,
                        'pagerank': score,
                        'rank': rank,
                        'section': section,
                        'keywords': keywords,
                        'primary_metric': primary_metric,
                        'found': True
                    })
                else:
                    priority_analysis.append({
                        'url': url,
                        'pagerank': 0,
                        'rank': 0,
                        'section': 'Unknown',
                        'keywords': keywords,
                        'primary_metric': primary_metric,
                        'found': False
                    })
        
        # Create enhanced visualizations
        visualizer = EnhancedVisualizer(pagerank_scores, section_mapping, section_analyzer, crawler.graph, crawler.page_data)
        
        # Display in enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 5 Critical Questions", "📊 Advanced Analytics", "🕸️ Network Graph", 
            "🤖 Enhanced AI Recommendations", "📋 CSV Export Hub", "🔍 Deep Insights"
        ])
        
        with tab1:
            st.markdown("## 🎯 The 5 Critical PageRank Questions - Enhanced Analysis")
            
            # Question 1 - Enhanced
            st.markdown('<div class="question-header">1. Which Sections receive the most PageRank? (Enhanced Business Intelligence)</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig_sections = visualizer.create_section_bar_chart()
                st.plotly_chart(fig_sections, use_container_width=True)
            
            with col2:
                st.markdown("### 🏆 Top Performing Sections")
                for i, stats in enumerate(section_stats[:6], 1):
                    business_colors = {
                        'critical': '🔴',
                        'high': '🟠', 
                        'medium': '🟡',
                        'low': '🟢',
                        'minimal': '⚫'
                    }
                    emoji = business_colors.get(stats['business_value'], '🔵')
                    
                    st.markdown(f"""
                    **{i}. {stats['section'].title()}** {emoji}  
                    📊 {stats['total_pr']:.4f} ({stats['percentage']:.1f}%)  
                    📄 {stats['page_count']} pages  
                    🎯 {stats['conversion_score']}/100 conversion potential
                    """)
                
                # Key insights
                critical_high_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] in ['critical', 'high'])
                efficiency_score = (critical_high_pr / total_pr * 100) if total_pr > 0 else 0
                
                if efficiency_score > 70:
                    st.markdown('<div class="success-card"><h4>✅ Excellent Distribution</h4><p>Most PageRank flows to high-value sections</p></div>', unsafe_allow_html=True)
                elif efficiency_score > 50:
                    st.markdown('<div class="warning-card"><h4>⚠️ Good but Improvable</h4><p>Opportunity to optimize PageRank flow</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-card"><h4>🚨 Needs Optimization</h4><p>Significant PageRank waste detected</p></div>', unsafe_allow_html=True)
            
            # Question 2 - Enhanced
            st.markdown('<div class="question-header">2. Which specific pages receive the most PageRank? (Performance Analysis)</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_top_pages = visualizer.create_top_pages_chart()
                st.plotly_chart(fig_top_pages, use_container_width=True)
            
            with col2:
                # Enhanced top pages analysis
                st.markdown("### 📊 Top Page Insights")
                
                top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for i, (url, score) in enumerate(top_pages[:5], 1):
                    page_data = crawler.page_data.get(url, {})
                    section = section_mapping.get(url, 'other')
                    business_value = section_analyzer.get_business_value(section)
                    
                    page_name = urlparse(url).path.split('/')[-1] or 'Homepage'
                    if len(page_name) > 20:
                        page_name = page_name[:17] + '...'
                    
                    color = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'minimal': '⚫'}[business_value]
                    
                    st.markdown(f"""
                    **{i}. {page_name}** {color}  
                    📊 {score:.6f} PageRank  
                    🏷️ {section} section  
                    📝 {page_data.get('word_count', 0):,} words
                    """)
            
            # Enhanced top pages table
            top_pages_data = []
            for i, (url, score) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20], 1):
                page_data = crawler.page_data.get(url, {})
                section = section_mapping.get(url, 'other')
                business_value = section_analyzer.get_business_value(section)
                conversion_score = section_analyzer.get_conversion_score(section)
                
                top_pages_data.append({
                    'Rank': i,
                    'URL': url[:70] + '...' if len(url) > 70 else url,
                    'PageRank': f"{score:.6f}",
                    'Section': section,
                    'Business Value': business_value,
                    'Conversion Score': f"{conversion_score}/100",
                    'Word Count': f"{page_data.get('word_count', 0):,}",
                    'Internal Links': page_data.get('internal_links', 0),
                    'Title': page_data.get('title', '')[:50] + '...' if len(page_data.get('title', '')) > 50 else page_data.get('title', '')
                })
            
            st.markdown("### 📋 Top 20 Pages Detailed Analysis")
            st.dataframe(pd.DataFrame(top_pages_data), use_container_width=True)
            
            # Question 3 - Enhanced
            st.markdown('<div class="question-header">3. Do these align with your Priority Target Pages? (Strategic Alignment)</div>', unsafe_allow_html=True)
            
            if priority_analysis:
                found_count = len([p for p in priority_analysis if p['found']])
                total_count = len(priority_analysis)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    coverage_pct = (found_count/total_count*100) if total_count > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Coverage Rate</h3>
                        <h2>{found_count}/{total_count}</h2>
                        <p>{coverage_pct:.1f}% found in crawl</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if found_count > 0:
                        avg_pr = np.mean([p['pagerank'] for p in priority_analysis if p['found']])
                        avg_rank = np.mean([p['rank'] for p in priority_analysis if p['found']])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Avg Performance</h3>
                            <h2>{avg_pr:.6f}</h2>
                            <p>Avg rank: {avg_rank:.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    top_priority_performers = len([p for p in priority_analysis if p['found'] and p['rank'] <= len(pagerank_scores) * 0.2])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🏆 Top Performers</h3>
                        <h2>{top_priority_performers}</h2>
                        <p>in top 20% by PageRank</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    missing_count = total_count - found_count
                    color = "success" if missing_count == 0 else "warning" if missing_count < total_count/2 else "critical"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚠️ Missing Pages</h3>
                        <h2>{missing_count}</h2>
                        <p>need internal links</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced priority pages analysis
                st.markdown("### 🎯 Priority Pages Performance Analysis")
                
                priority_performance_data = []
                for p in priority_analysis:
                    status = "✅ Excellent" if p['found'] and p['rank'] <= len(pagerank_scores) * 0.1 else \
                            "🟡 Good" if p['found'] and p['rank'] <= len(pagerank_scores) * 0.3 else \
                            "🟠 Needs Improvement" if p['found'] else \
                            "🔴 Missing"
                    
                    priority_performance_data.append({
                        'URL': p['url'][:60] + '...' if len(p['url']) > 60 else p['url'],
                        'PageRank Score': f"{p['pagerank']:.6f}" if p['found'] else 'Not Found',
                        'Rank': f"{p['rank']}" if p['found'] else 'N/A',
                        'Percentile': f"{(p['rank']/len(pagerank_scores)*100):.1f}%" if p['found'] else 'N/A',
                        'Section': p['section'],
                        'Status': status,
                        'Keywords': p['keywords'][:40] + '...' if len(str(p['keywords'])) > 40 else p['keywords'],
                        'Primary Metric': p.get('primary_metric', 'N/A')
                    })
                
                st.dataframe(pd.DataFrame(priority_performance_data), use_container_width=True)
                
                # Alignment insights
                if coverage_pct < 80:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h4>🚨 Priority Page Coverage Issue</h4>
                        <p><strong>{missing_count} priority pages</strong> were not found in the crawl. This suggests:</p>
                        <ul>
                            <li>Missing internal links to these pages</li>
                            <li>Pages may be too deep in site structure</li>
                            <li>Possible crawl limitations or restricted access</li>
                            <li>Pages might not exist or have changed URLs</li>
                        </ul>
                        <p><strong>Recommendation:</strong> Improve internal linking to missing priority pages.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="insight-card">
                    <h4>📁 Upload Priority Pages for Analysis</h4>
                    <p>Upload a CSV file with your priority pages to get detailed alignment analysis. The file can include:</p>
                    <ul>
                        <li>Target landing pages</li>
                        <li>High-converting pages</li>
                        <li>Strategic content pages</li>
                        <li>Pages with SEO targets</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Question 4 - Enhanced
            st.markdown('<div class="question-header">4. How can we reduce PageRank to non-valuable sections? (Optimization Strategy)</div>', unsafe_allow_html=True)
            
            # Calculate waste metrics
            low_value_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] in ['low', 'minimal'])
            waste_percentage = (low_value_pr / total_pr * 100) if total_pr > 0 else 0
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                color_class = "success" if waste_percentage < 10 else "warning" if waste_percentage < 20 else "warning"
                st.markdown(f"""
                <div class="{color_class}-card">
                    <h3>🔴 PageRank Waste Analysis</h3>
                    <h2>{waste_percentage:.1f}%</h2>
                    <p>flowing to low-value sections</p>
                    <p><strong>Potential Recovery:</strong> {low_value_pr:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Waste breakdown
                st.markdown("### 📊 Waste Breakdown")
                waste_sections = [s for s in section_stats if s['business_value'] in ['low', 'minimal']]
                for section in waste_sections[:5]:
                    st.markdown(f"• **{section['section']}**: {section['percentage']:.1f}% ({section['page_count']} pages)")
            
            with col2:
                fig_business = visualizer.create_business_value_pie()
                st.plotly_chart(fig_business, use_container_width=True)
            
            # Optimization recommendations
            st.markdown("### 🛠️ PageRank Waste Reduction Strategies")
            
            optimization_strategies = []
            
            # Find high-PR low-value pages
            low_value_high_pr = []
            for url, score in pagerank_scores.items():
                section = section_mapping.get(url, 'other')
                business_value = section_analyzer.get_business_value(section)
                if business_value in ['low', 'minimal'] and score > np.percentile(list(pagerank_scores.values()), 70):
                    low_value_high_pr.append((url, score, section))
            
            if low_value_high_pr:
                optimization_strategies.append({
                    'title': 'Redirect High-PR Low-Value Pages',
                    'description': f'Found {len(low_value_high_pr)} low-value pages in top 30% by PageRank',
                    'impact': 'High',
                    'effort': 'Medium',
                    'details': f'These pages are receiving {sum(score for _, score, _ in low_value_high_pr):.4f} PageRank that could be redirected'
                })
            
            # Analyze internal linking patterns
            section_linking = defaultdict(lambda: defaultdict(int))
            for source, target in crawler.graph.edges():
                source_section = section_mapping.get(source, 'other')
                target_section = section_mapping.get(target, 'other')
                section_linking[source_section][target_section] += 1
            
            # Find sections linking heavily to low-value sections
            for source_section, targets in section_linking.items():
                total_links = sum(targets.values())
                low_value_links = sum(count for target_section, count in targets.items() 
                                    if section_analyzer.get_business_value(target_section) in ['low', 'minimal'])
                
                if total_links >= 10 and (low_value_links / total_links) > 0.25:
                    optimization_strategies.append({
                        'title': f'Optimize {source_section.title()} Section Linking',
                        'description': f'{(low_value_links/total_links*100):.1f}% of links flow to low-value sections',
                        'impact': 'Medium',
                        'effort': 'Low',
                        'details': f'Reduce {low_value_links} links to low-value sections, redirect to high-value pages'
                    })
            
            # Content consolidation opportunities
            thin_content_pages = [url for url, data in crawler.page_data.items() 
                                if data.get('word_count', 0) < 300 and pagerank_scores.get(url, 0) > 0.001]
            
            if len(thin_content_pages) > 5:
                optimization_strategies.append({
                    'title': 'Consolidate Thin Content Pages',
                    'description': f'Found {len(thin_content_pages)} thin content pages with PageRank',
                    'impact': 'Medium',
                    'effort': 'High',
                    'details': 'Merge thin pages into comprehensive resources to concentrate PageRank'
                })
            
            # Display strategies
            if optimization_strategies:
                for i, strategy in enumerate(optimization_strategies, 1):
                    impact_color = {'High': 'priority-high', 'Medium': 'priority-medium', 'Low': 'priority-low'}[strategy['impact']]
                    st.markdown(f"""
                    <div class="recommendation-section {impact_color}">
                        <h4>{i}. {strategy['title']} ({strategy['impact']} Impact, {strategy['effort']} Effort)</h4>
                        <p><strong>Issue:</strong> {strategy['description']}</p>
                        <p><strong>Action:</strong> {strategy['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card"><h4>✅ Excellent PageRank Distribution</h4><p>No major waste issues detected. Your PageRank is flowing efficiently!</p></div>', unsafe_allow_html=True)
            
            # Question 5 - Enhanced
            st.markdown('<div class="question-header">5. How can we redirect PageRank to priority pages? (Strategic Implementation)</div>', unsafe_allow_html=True)
            
            # Advanced PageRank flow analysis
            st.markdown("### 🎯 Strategic PageRank Redistribution Plan")
            
            redistribution_strategies = []
            
            # High-authority pages that could link to priority pages
            if priority_analysis:
                priority_urls = {p['url'] for p in priority_analysis if p['found']}
                high_authority_pages = [url for url, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:30]
                                      if url not in priority_urls]
                
                # Calculate potential impact for each linking opportunity
                linking_opportunities = []
                
                for high_auth_url in high_authority_pages[:10]:
                    high_auth_section = section_mapping.get(high_auth_url, 'other')
                    high_auth_score = pagerank_scores[high_auth_url]
                    
                    for priority_page in priority_analysis[:10]:
                        if priority_page['found']:
                            priority_url = priority_page['url']
                            priority_section = section_mapping.get(priority_url, 'other')
                            current_pr = priority_page['pagerank']
                            
                            # Estimate PageRank boost (simplified calculation)
                            potential_boost = high_auth_score * 0.85 * 0.15  # Damping factor * link value
                            impact_percentage = (potential_boost / current_pr * 100) if current_pr > 0 else 0
                            
                            if impact_percentage > 5:  # Only show significant opportunities
                                linking_opportunities.append({
                                    'source_url': high_auth_url,
                                    'source_section': high_auth_section,
                                    'source_pr': high_auth_score,
                                    'target_url': priority_url,
                                    'target_section': priority_section,
                                    'current_pr': current_pr,
                                    'potential_boost': potential_boost,
                                    'impact_percentage': impact_percentage
                                })
                
                if linking_opportunities:
                    # Sort by impact
                    linking_opportunities.sort(key=lambda x: x['impact_percentage'], reverse=True)
                    
                    st.markdown("### 🔗 Top Linking Opportunities")
                    
                    opportunity_data = []
                    for i, opp in enumerate(linking_opportunities[:15], 1):
                        opportunity_data.append({
                            'Rank': i,
                            'Source Page': opp['source_url'][:50] + '...' if len(opp['source_url']) > 50 else opp['source_url'],
                            'Source PR': f"{opp['source_pr']:.6f}",
                            'Source Section': opp['source_section'],
                            'Target Page': opp['target_url'][:50] + '...' if len(opp['target_url']) > 50 else opp['target_url'],
                            'Target Section': opp['target_section'],
                            'Current Target PR': f"{opp['current_pr']:.6f}",
                            'Potential Boost': f"{opp['potential_boost']:.6f}",
                            'Impact %': f"{opp['impact_percentage']:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(opportunity_data), use_container_width=True)
                    
                    # Implementation guidance
                    top_opportunity = linking_opportunities[0]
                    st.markdown(f"""
                    <div class="recommendation-section priority-high">
                        <h4>🎯 #1 Priority Implementation</h4>
                        <p><strong>Add contextual link from:</strong></p>
                        <p>📄 {top_opportunity['source_url'][:60]}... ({top_opportunity['source_section']} section)</p>
                        <p><strong>To:</strong></p>
                        <p>🎯 {top_opportunity['target_url'][:60]}... ({top_opportunity['target_section']} section)</p>
                        <p><strong>Expected Impact:</strong> {top_opportunity['impact_percentage']:.1f}% PageRank increase</p>
                        <p><strong>Implementation:</strong> Add relevant anchor text link in main content area</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # General redistribution strategies
            general_strategies = [
                {
                    'title': 'Optimize Main Navigation',
                    'description': 'Ensure high-priority pages are prominently featured in main navigation',
                    'impact': 'High',
                    'implementation': 'Add priority pages to header/footer navigation with strategic anchor text'
                },
                {
                    'title': 'Create Content Hub Structure',
                    'description': 'Develop topic clusters that naturally link to priority pages',
                    'impact': 'High',
                    'implementation': 'Build pillar content that extensively links to related priority pages'
                },
                {
                    'title': 'Implement Contextual Cross-Linking',
                    'description': 'Add relevant internal links within existing content',
                    'impact': 'Medium',
                    'implementation': 'Audit content for natural linking opportunities to priority pages'
                },
                {
                    'title': 'Optimize Breadcrumb Structure',
                    'description': 'Ensure logical hierarchical linking to important pages',
                    'impact': 'Medium',
                    'implementation': 'Review and enhance breadcrumb navigation for better PageRank flow'
                }
            ]
            
            st.markdown("### 🚀 General Redistribution Strategies")
            
            for i, strategy in enumerate(general_strategies, 1):
                impact_color = {'High': 'priority-high', 'Medium': 'priority-medium', 'Low': 'priority-low'}[strategy['impact']]
                st.markdown(f"""
                <div class="recommendation-section {impact_color}">
                    <h4>{i}. {strategy['title']} ({strategy['impact']} Impact)</h4>
                    <p><strong>Strategy:</strong> {strategy['description']}</p>
                    <p><strong>Implementation:</strong> {strategy['implementation']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("## 📊 Advanced Analytics & Deep Insights")
            
            # Section performance heatmap
            st.markdown("### 🔥 Section Linking Patterns")
            fig_heatmap = visualizer.create_section_matrix_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Route depth analysis with enhanced insights
            st.markdown("### 📊 URL Structure & Depth Analysis")
            
            depth_data = defaultdict(list)
            for url, data in crawler.page_data.items():
                depth = data.get('route_depth', 0)
                pr_score = pagerank_scores.get(url, 0)
                depth_data[depth].append(pr_score)
            
            if depth_data:
                depth_analysis = []
                for depth, scores in depth_data.items():
                    if scores:
                        depth_analysis.append({
                            'depth': depth,
                            'page_count': len(scores),
                            'total_pagerank': sum(scores),
                            'avg_pagerank': np.mean(scores),
                            'max_pagerank': max(scores),
                            'min_pagerank': min(scores)
                        })
                
                depth_df = pd.DataFrame(depth_analysis)
                
                if not depth_df.empty:
                    fig_depth = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Pages by Depth', 'Total PageRank by Depth', 
                                      'Average PageRank by Depth', 'PageRank Range by Depth'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Pages by depth
                    fig_depth.add_trace(
                        go.Bar(x=depth_df['depth'], y=depth_df['page_count'], 
                              name='Page Count', marker_color='#3b82f6'),
                        row=1, col=1
                    )
                    
                    # Total PageRank by depth
                    fig_depth.add_trace(
                        go.Bar(x=depth_df['depth'], y=depth_df['total_pagerank'],
                              name='Total PageRank', marker_color='#10b981'),
                        row=1, col=2
                    )
                    
                    # Average PageRank by depth
                    fig_depth.add_trace(
                        go.Scatter(x=depth_df['depth'], y=depth_df['avg_pagerank'],
                                  mode='lines+markers', name='Avg PageRank',
                                  line=dict(color='#f59e0b', width=3)),
                        row=2, col=1
                    )
                    
                    # PageRank range by depth
                    fig_depth.add_trace(
                        go.Scatter(x=depth_df['depth'], y=depth_df['max_pagerank'],
                                  mode='lines+markers', name='Max PageRank',
                                  line=dict(color='#ef4444')),
                        row=2, col=2
                    )
                    fig_depth.add_trace(
                        go.Scatter(x=depth_df['depth'], y=depth_df['min_pagerank'],
                                  mode='lines+markers', name='Min PageRank',
                                  line=dict(color='#6b7280')),
                        row=2, col=2
                    )
                    
                    fig_depth.update_layout(height=600, showlegend=False,
                                          title_text="📈 Comprehensive Route Depth Analysis")
                    
                    st.plotly_chart(fig_depth, use_container_width=True)
                    
                    # Depth insights
                    if len(depth_df) > 1:
                        optimal_depth = depth_df.loc[depth_df['avg_pagerank'].idxmax(), 'depth']
                        deep_pages = depth_df[depth_df['depth'] > 3]['page_count'].sum() if len(depth_df[depth_df['depth'] > 3]) > 0 else 0
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="insight-card">
                                <h4>🎯 Optimal Depth Analysis</h4>
                                <p><strong>Best performing depth:</strong> Level {optimal_depth}</p>
                                <p><strong>Deep pages (>3 levels):</strong> {deep_pages}</p>
                                <p><strong>Recommendation:</strong> {'Consider promoting deep pages' if deep_pages > 0 else 'Good depth distribution'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Show depth distribution table
                            st.markdown("**📊 Depth Distribution:**")
                            st.dataframe(depth_df[['depth', 'page_count', 'avg_pagerank']].round(6), use_container_width=True)
            
            # Content quality analysis
            st.markdown("### 📝 Content Quality vs PageRank Analysis")
            
            content_quality_data = []
            for url, pr_score in pagerank_scores.items():
                page_info = crawler.page_data.get(url, {})
                word_count = page_info.get('word_count', 0)
                internal_links = page_info.get('internal_links', 0)
                section = section_mapping.get(url, 'other')
                business_value = section_analyzer.get_business_value(section)
                
                # Quality score calculation
                quality_score = 0
                if word_count > 1000:
                    quality_score += 3
                elif word_count > 500:
                    quality_score += 2
                elif word_count > 200:
                    quality_score += 1
                
                if internal_links > 10:
                    quality_score += 2
                elif internal_links > 5:
                    quality_score += 1
                
                if page_info.get('title') and len(page_info.get('title', '')) > 30:
                    quality_score += 1
                
                if page_info.get('meta_description') and len(page_info.get('meta_description', '')) > 120:
                    quality_score += 1
                
                content_quality_data.append({
                    'url': url,
                    'pagerank': pr_score,
                    'word_count': word_count,
                    'internal_links': internal_links,
                    'quality_score': quality_score,
                    'business_value': business_value,
                    'section': section
                })
            
            quality_df = pd.DataFrame(content_quality_data)
            
            if not quality_df.empty:
                # Create scatter plot
                fig_quality = px.scatter(
                    quality_df,
                    x='quality_score',
                    y='pagerank',
                    color='business_value',
                    size='word_count',
                    hover_data=['section', 'internal_links'],
                    color_discrete_map={
                        'critical': '#dc2626',
                        'high': '#ea580c',
                        'medium': '#ca8a04',
                        'low': '#16a34a',
                        'minimal': '#9ca3af'
                    },
                    title='📊 Content Quality vs PageRank Correlation',
                    labels={'quality_score': 'Content Quality Score (0-8)', 'pagerank': 'PageRank Score'}
                )
                
                fig_quality.update_layout(height=500)
                st.plotly_chart(fig_quality, use_container_width=True)
                
                # Quality insights
                correlation = quality_df['quality_score'].corr(quality_df['pagerank'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📈 Quality-PageRank Correlation</h3>
                        <h2>{correlation:.3f}</h2>
                        <p>{'Strong positive' if correlation > 0.5 else 'Moderate' if correlation > 0.3 else 'Weak'} correlation</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    high_quality_low_pr = len(quality_df[(quality_df['quality_score'] >= 6) & (quality_df['pagerank'] < np.percentile(quality_df['pagerank'], 50))])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💎 Hidden Gems</h3>
                        <h2>{high_quality_low_pr}</h2>
                        <p>High quality, low PageRank pages</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("## 🕸️ Interactive Network Visualization")
            
            st.markdown("""
            **Enhanced Network Analysis** - This visualization shows the top 30 pages and their PageRank flow relationships:
            
            🔴 **Critical Value** | 🟠 **High Value** | 🟡 **Medium Value** | 🟢 **Low Value** | ⚫ **Minimal Value**
            
            - **Node size** = PageRank score (larger = higher PageRank)
            - **Node color** = Business value classification  
            - **Lines** = Internal link connections between pages
            - **Hover** = Detailed page information
            """)
            
            # Create enhanced network graph
            fig_network = visualizer.create_network_graph()
            
            if fig_network.data:
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Network analysis insights
                st.markdown("### 📊 Network Intelligence")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Calculate network metrics
                    top_30_urls = [url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:30]]
                    subgraph = crawler.graph.subgraph(top_30_urls)
                    node_count = len(subgraph.nodes())
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔗 Network Nodes</h3>
                        <h2>{node_count}</h2>
                        <p>Top pages displayed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    edge_count = len(subgraph.edges())
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>↔️ Connections</h3>
                        <h2>{edge_count}</h2>
                        <p>Internal links shown</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Calculate network density
                    if node_count > 1:
                        density = nx.density(subgraph) * 100
                    else:
                        density = 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🕸️ Network Density</h3>
                        <h2>{density:.1f}%</h2>
                        <p>Interconnectedness</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Calculate centralization
                    try:
                        if node_count > 2:
                            centrality_scores = nx.degree_centrality(subgraph)
                            avg_centrality = np.mean(list(centrality_scores.values())) * 100
                        else:
                            avg_centrality = 0
                    except:
                        avg_centrality = 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Avg Centrality</h3>
                        <h2>{avg_centrality:.1f}%</h2>
                        <p>Distribution score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Network insights
                if density < 10:
                    st.markdown("""
                    <div class="warning-card">
                        <h4>⚠️ Low Network Density Detected</h4>
                        <p>Your top pages have limited interconnections. Consider adding more contextual internal links between high-performing pages to improve PageRank flow.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif density > 30:
                    st.markdown("""
                    <div class="success-card">
                        <h4>✅ Excellent Network Connectivity</h4>
                        <p>Your top pages are well-connected, facilitating good PageRank distribution. Maintain this linking strategy!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ No network data available for visualization. This may occur if no internal links were found.")
        
        with tab4:
            st.markdown("## 🤖 Enhanced AI-Powered Strategic Recommendations")
            
            if openai_key:
                if st.button("🧠 Generate Comprehensive AI Analysis", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is conducting deep analysis of your PageRank data..."):
                        ai_recommendations = generate_enhanced_ai_recommendations(
                            pagerank_scores, section_stats, priority_analysis, 
                            crawler.page_data, crawler.graph, openai_key, website_url
                        )
                        
                        st.markdown(f"""
                        <div class="ai-card">
                            <h3>🤖 Comprehensive AI Strategic Analysis</h3>
                            <div style="white-space: pre-wrap; line-height: 1.7; font-size: 14px;">{ai_recommendations}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional AI-powered insights
                st.markdown("### 🎯 Automated Strategic Insights")
                
                insights = []
                
                # Calculate advanced metrics for insights
                efficiency_score = sum(s['total_pr'] for s in section_stats if s['business_value'] in ['critical', 'high']) / total_pr * 100 if total_pr > 0 else 0
                waste_percentage = sum(s['total_pr'] for s in section_stats if s['business_value'] in ['low', 'minimal']) / total_pr * 100 if total_pr > 0 else 0
                
                if waste_percentage > 20:
                    insights.append(("warning", "🚨 Critical PageRank Waste", f"High waste detected: {waste_percentage:.1f}% flows to low-value sections. Immediate action required."))
                elif waste_percentage > 15:
                    insights.append(("warning", "⚠️ Significant PageRank Waste", f"Moderate waste: {waste_percentage:.1f}% flows to low-value sections. Optimization recommended."))
                
                if efficiency_score > 75:
                    insights.append(("success", "✅ Excellent PageRank Efficiency", f"Outstanding distribution: {efficiency_score:.1f}% flows to high-value sections."))
                elif efficiency_score > 60:
                    insights.append(("success", "👍 Good PageRank Efficiency", f"Solid distribution: {efficiency_score:.1f}% flows to high-value sections."))
                else:
                    insights.append(("warning", "📈 PageRank Efficiency Opportunity", f"Room for improvement: Only {efficiency_score:.1f}% flows to high-value sections."))
                
                # Network analysis insights
                network_density = nx.density(crawler.graph) * 100 if len(crawler.graph.nodes()) > 1 else 0
                if network_density < 5:
                    insights.append(("warning", "🕸️ Sparse Internal Linking", f"Low network density ({network_density:.1f}%) suggests insufficient internal linking."))
                elif network_density > 20:
                    insights.append(("success", "🔗 Strong Internal Linking", f"Excellent network density ({network_density:.1f}%) indicates good internal linking strategy."))
                
                # Content analysis insights
                if crawler.page_data:
                    avg_word_count = np.mean([data.get('word_count', 0) for data in crawler.page_data.values()])
                    if avg_word_count < 400:
                        insights.append(("warning", "📝 Thin Content Alert", f"Average page length is only {avg_word_count:.0f} words. Consider expanding content."))
                    elif avg_word_count > 1000:
                        insights.append(("success", "📚 Rich Content Strategy", f"Excellent average page length of {avg_word_count:.0f} words supports SEO goals."))
                
                # Priority page insights
                if priority_analysis:
                    found_count = len([p for p in priority_analysis if p['found']])
                    coverage_rate = found_count / len(priority_analysis) * 100
                    
                    if coverage_rate < 70:
                        insights.append(("warning", "🎯 Priority Page Coverage Gap", f"Only {coverage_rate:.1f}% of priority pages found in crawl. Improve internal linking."))
                    elif coverage_rate > 90:
                        insights.append(("success", "🎯 Excellent Priority Page Coverage", f"Outstanding {coverage_rate:.1f}% of priority pages found in crawl."))
                
                # Display insights
                for card_type, title, description in insights:
                    st.markdown(f"""
                    <div class="{card_type}-card">
                        <h4>{title}</h4>
                        <p>{description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="ai-card">
                    <h3>🤖 Enhanced AI Recommendations Available</h3>
                    <p><strong>Unlock Advanced AI Analysis:</strong></p>
                    <ul>
                        <li>🎯 Strategic implementation roadmap with timelines</li>
                        <li>📊 Quantified impact projections and ROI estimates</li>
                        <li>🛠️ Technical implementation guidance</li>
                        <li>📈 Expected traffic and conversion improvements</li>
                        <li>🔍 Advanced competitive insights</li>
                        <li>🚀 Long-term optimization strategy</li>
                    </ul>
                    <p><em>Add your OpenAI API key in the sidebar to access comprehensive AI-powered recommendations.</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("## 📋 Comprehensive CSV Export Hub")
            
            st.markdown("""
            <div class="insight-card">
                <h3>📊 Multi-Report CSV Export System</h3>
                <p>Generate comprehensive CSV reports for detailed analysis, team sharing, and data integration:</p>
                <ul>
                    <li><strong>Executive Summary:</strong> High-level metrics and KPIs</li>
                    <li><strong>Detailed Pages Analysis:</strong> Complete page-by-page breakdown</li>
                    <li><strong>Section Performance:</strong> Section-level insights and recommendations</li>
                    <li><strong>Priority Pages Analysis:</strong> Performance of your key pages</li>
                    <li><strong>Internal Linking Opportunities:</strong> Strategic linking recommendations</li>
                    <li><strong>Technical SEO Issues:</strong> Page-level technical problems</li>
                </ul>
            </div>
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("📊 Generate Complete CSV Report Package", type="primary", use_container_width=True):
                    with st.spinner("📋 Generating comprehensive CSV reports..."):
                        reports, timestamp = generate_comprehensive_csv_reports(
                            crawler, pagerank_scores, section_mapping, section_analyzer, priority_df
                        )
                        
                        if reports:
                            st.success("✅ CSV reports generated successfully!")
                            
                            # Create download buttons for each report
                            for report_name, report_df in reports.items():
                                if not report_df.empty:
                                    csv_data = report_df.to_csv(index=False)
                                    filename = f"pagerank_{report_name}_{timestamp}.csv"
                                    
                                    # Format report name for display
                                    display_name = report_name.replace('_', ' ').title()
                                    
                                    st.download_button(
                                        label=f"📥 Download {display_name} Report",
                                        data=csv_data,
                                        file_name=filename,
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            # Show report previews
                            st.markdown("### 📊 Report Previews")
                            
                            for report_name, report_df in reports.items():
                                if not report_df.empty:
                                    display_name = report_name.replace('_', ' ').title()
                                    
                                    with st.expander(f"📋 {display_name} Preview ({len(report_df)} rows)"):
                                        st.dataframe(report_df.head(10), use_container_width=True)
                                        
                                        # Show column info
                                        st.markdown(f"**Columns:** {', '.join(report_df.columns.tolist())}")
                        else:
                            st.error("❌ Failed to generate CSV reports. Please try again.")
            
            with col2:
                st.markdown("### 📊 Report Statistics")
                
                if 'reports' in locals() and reports:
                    for report_name, report_df in reports.items():
                        if not report_df.empty:
                            display_name = report_name.replace('_', ' ').title()
                            file_size = len(report_df.to_csv(index=False)) / 1024  # KB
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{display_name}</h4>
                                <p><strong>Rows:</strong> {len(report_df):,}</p>
                                <p><strong>Columns:</strong> {len(report_df.columns)}</p>
                                <p><strong>Size:</strong> {file_size:.1f} KB</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Click 'Generate Reports' to see statistics")
            
            # JSON export option
            st.markdown("---")
            st.markdown("### 📄 JSON Export Option")
            
            if st.button("📄 Generate JSON Data Export"):
                comprehensive_data = {
                    'analysis_metadata': {
                        'website_url': website_url,
                        'analysis_date': datetime.now().isoformat(),
                        'total_pages': len(pagerank_scores),
                        'total_links': len(crawler.graph.edges()),
                        'sections_identified': len(set(section_mapping.values())),
                        'crawl_duration': crawl_duration,
                        'efficiency_score': efficiency_score,
                        'waste_percentage': waste_percentage
                    },
                    'pagerank_scores': {url: float(score) for url, score in pagerank_scores.items()},
                    'section_mapping': section_mapping,
                    'section_statistics': section_stats,
                    'page_data': crawler.page_data,
                    'priority_analysis': priority_analysis or [],
                    'network_metrics': {
                        'density': nx.density(crawler.graph) * 100 if len(crawler.graph.nodes()) > 1 else 0,
                        'node_count': len(crawler.graph.nodes()),
                        'edge_count': len(crawler.graph.edges())
                    }
                }
                
                json_data = json.dumps(comprehensive_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Complete JSON Export",
                    data=json_data,
                    file_name=f"pagerank_complete_analysis_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with tab6:
            st.markdown("## 🔍 Deep Performance Insights")
            
            # Advanced performance metrics
            st.markdown("### 🎯 Advanced Performance Metrics")
            
            # Calculate advanced KPIs
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                # PageRank concentration (Gini coefficient)
                pr_values = sorted(pagerank_scores.values(), reverse=True)
                if len(pr_values) > 1:
                    n = len(pr_values)
                    cumsum = np.cumsum(pr_values)
                    gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(pr_values))) / (n * sum(pr_values))
                    concentration_score = gini * 100
                else:
                    concentration_score = 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 PR Concentration</h3>
                    <h2>{concentration_score:.1f}</h2>
                    <p>Gini coefficient (%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with kpi_col2:
                # Content efficiency (PR per word)
                if crawler.page_data:
                    total_words = sum(data.get('word_count', 0) for data in crawler.page_data.values())
                    content_efficiency = total_pr / total_words * 1000000 if total_words > 0 else 0
                else:
                    content_efficiency = 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Content Efficiency</h3>
                    <h2>{content_efficiency:.2f}</h2>
                    <p>PR per 1M words</p>
                </div>
                """, unsafe_allow_html=True)
            
            with kpi_col3:
                # Link equity distribution
                if len(crawler.graph.edges()) > 0:
                    total_links = len(crawler.graph.edges())
                    link_equity = total_pr / total_links
                else:
                    link_equity = 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔗 Link Equity</h3>
                    <h2>{link_equity:.6f}</h2>
                    <p>PR per internal link</p>
                </div>
                """, unsafe_allow_html=True)
            
            with kpi_col4:
                # Authority distribution
                top_10_pr = sum(score for _, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10])
                authority_concentration = (top_10_pr / total_pr * 100) if total_pr > 0 else 0
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👑 Authority Focus</h3>
                    <h2>{authority_concentration:.1f}%</h2>
                    <p>Top 10 pages share</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance recommendations based on metrics
            st.markdown("### 🎯 Performance Optimization Recommendations")
            
            recommendations = []
            
            if concentration_score > 70:
                recommendations.append({
                    'type': 'warning',
                    'title': 'High PageRank Concentration',
                    'description': f'PageRank is highly concentrated ({concentration_score:.1f}). Consider distributing authority more evenly.',
                    'action': 'Implement hub-and-spoke linking from high-authority pages to important content.'
                })
            elif concentration_score < 30:
                recommendations.append({
                    'type': 'warning',
                    'title': 'PageRank Too Distributed',
                    'description': f'PageRank may be too evenly distributed ({concentration_score:.1f}). Focus authority on key pages.',
                    'action': 'Consolidate authority by linking more strategically to your most important pages.'
                })
            
            if content_efficiency < 1:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Low Content Efficiency',
                    'description': f'Content efficiency is low ({content_efficiency:.2f}). Pages may have thin content or poor linking.',
                    'action': 'Expand content on high-PageRank pages and improve internal linking strategy.'
                })
            
            if authority_concentration > 80:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Authority Too Concentrated',
                    'description': f'Top 10 pages hold {authority_concentration:.1f}% of PageRank. This may limit overall site authority.',
                    'action': 'Distribute authority to more pages through strategic internal linking.'
                })
            
            if link_equity < 0.00001:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Poor Link Equity Distribution',
                    'description': f'Link equity per internal link is low ({link_equity:.6f}). Links may not be effectively transferring authority.',
                    'action': 'Audit internal links and remove unnecessary or low-value links to concentrate equity.'
                })
            
            # Display recommendations
            for rec in recommendations:
                st.markdown(f"""
                <div class="{rec['type']}-card">
                    <h4>{rec['title']}</h4>
                    <p><strong>Issue:</strong> {rec['description']}</p>
                    <p><strong>Action:</strong> {rec['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if not recommendations:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Excellent Performance Metrics</h4>
                    <p>Your PageRank distribution shows healthy performance across all key metrics. Continue monitoring and maintain current strategies.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Final summary
        st.markdown("---")
        st.markdown("## 🎉 Analysis Complete!")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 Total Pages</h3>
                <h2>{len(pagerank_scores):,}</h2>
                <p>analyzed successfully</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔗 Internal Links</h3>
                <h2>{len(crawler.graph.edges()):,}</h2>
                <p>discovered and mapped</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏷️ Sections</h3>
                <h2>{len(set(section_mapping.values()))}</h2>
                <p>intelligently categorized</p>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Efficiency</h3>
                <h2>{efficiency_score:.1f}%</h2>
                <p>PageRank optimization score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-card">
            <h3>🚀 Next Steps for Maximum Impact</h3>
            <ol>
                <li><strong>Download CSV reports</strong> for detailed analysis and team sharing</li>
                <li><strong>Implement AI recommendations</strong> starting with highest-impact changes</li>
                <li><strong>Focus on internal linking</strong> between high-authority and priority pages</li>
                <li><strong>Monitor progress</strong> by re-running analysis after optimizations</li>
                <li><strong>Integrate findings</strong> with your broader SEO and content strategy</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

