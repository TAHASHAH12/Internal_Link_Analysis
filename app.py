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
from openai import OpenAI
import os
import gc
import logging
import io
import csv

# Configure logging
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
    .main > div {
        padding-top: 2rem;
    }
    
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .five-questions-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .question-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .heatmap-cell {
        border: 1px solid #ddd;
        padding: 0.5rem;
        text-align: center;
        min-height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .anchor-text-box {
        background: #e9ecef;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .priority-highlight {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .low-performance {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .linking-opportunity {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .section-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #4dabf7 0%, #69db7c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .crawl-progress {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

###############################################################################
#                         Utility Functions & Classes                         #
###############################################################################

def fix_dataframe_types(df):
    """Fix DataFrame types to prevent Arrow serialization errors"""
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].dtype == 'object':
            df_copy[column] = df_copy[column].astype(str)
        elif df_copy[column].dtype in ['int64', 'float64']:
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    return df_copy

def safe_iloc_access(df, index, column=None):
    """Safe access to DataFrame iloc to prevent indexing errors"""
    try:
        if column is None:
            return df.iloc[index] if isinstance(index, int) else df.loc[index]
        
        if isinstance(index, int):
            if isinstance(column, str):
                return df.iloc[index][column]
            elif isinstance(column, int):
                return df.iloc[index, column]
        return df.loc[index, column]
    except (IndexError, KeyError, TypeError, ValueError):
        return 0 if column else pd.Series()

def safe_series_access(series, index):
    """Safe access to Series values"""
    try:
        return series.iloc[index] if isinstance(index, int) else series[index]
    except (IndexError, KeyError, TypeError):
        return 0

def categorize_url_by_structure(url):
    """Categorize URLs based on folder structure as requested"""
    parsed = urlparse(url)
    path = parsed.path.lower().strip('/')
    
    if not path:
        return 'homepage'
    
    # Split by forward slashes to create categories
    path_segments = [seg for seg in path.split('/') if seg]
    
    if not path_segments:
        return 'homepage'
    
    # Use first meaningful segment
    first_segment = path_segments[0]
    
    # Handle common patterns
    if first_segment.isdigit() or len(first_segment) < 2:
        if len(path_segments) > 1:
            first_segment = path_segments[1]
        else:
            first_segment = 'content'
    
    # Clean up the segment
    category = first_segment.replace('-', ' ').replace('_', ' ').title()
    
    return category

def csv_column_mapper(df, required_columns, prefix=""):
    """UI component for mapping CSV columns"""
    st.markdown("### 🔗 Column Mapping")
    mapping = {}
    
    for col in required_columns:
        available_columns = list(df.columns)
        # Try to auto-match columns
        best_match = None
        for available_col in available_columns:
            if col.lower() in available_col.lower():
                best_match = available_col
                break
        
        default_index = available_columns.index(best_match) if best_match else 0
        
        mapping[col] = st.selectbox(
            f"Select column for '{col}':",
            available_columns,
            index=default_index,
            key=f"{prefix}_{col}_mapping"
        )
    
    return mapping

def load_and_process_priority_csv(uploaded_file):
    """Load and process priority pages CSV with column mapping"""
    try:
        df = pd.read_csv(uploaded_file)
        
        required_columns = ['url', 'category', 'keyword']
        column_mapping = csv_column_mapper(df, required_columns, "priority")
        
        # Create mapped DataFrame
        mapped_df = pd.DataFrame()
        for req_col, selected_col in column_mapping.items():
            if selected_col in df.columns:
                mapped_df[req_col] = df[selected_col].astype(str)
            else:
                mapped_df[req_col] = ""
        
        return fix_dataframe_types(mapped_df)
    
    except Exception as e:
        st.error(f"Error processing priority pages CSV: {e}")
        return pd.DataFrame(columns=['url', 'category', 'keyword'])

def load_screaming_frog_csv(uploaded_file):
    """Load and process Screaming Frog CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Common Screaming Frog columns
        sf_columns = ['url', 'title', 'meta_description', 'h1', 'word_count', 'links_in']
        column_mapping = csv_column_mapper(df, sf_columns, "screaming_frog")
        
        # Process the data
        processed_data = {}
        graph = nx.DiGraph()
        anchor_texts = defaultdict(Counter)
        
        for _, row in df.iterrows():
            url = str(row[column_mapping['url']])
            if pd.isna(url) or url == 'nan':
                continue
                
            processed_data[url] = {
                'title': str(row.get(column_mapping.get('title', ''), '')),
                'meta_description': str(row.get(column_mapping.get('meta_description', ''), '')),
                'h1': str(row.get(column_mapping.get('h1', ''), '')),
                'word_count': int(row.get(column_mapping.get('word_count', ''), 0)) if pd.notna(row.get(column_mapping.get('word_count', ''), 0)) else 0,
                'internal_links': int(row.get(column_mapping.get('links_in', ''), 0)) if pd.notna(row.get(column_mapping.get('links_in', ''), 0)) else 0,
            }
            
            graph.add_node(url)
        
        return processed_data, graph, anchor_texts
    
    except Exception as e:
        st.error(f"Error processing Screaming Frog CSV: {e}")
        return {}, nx.DiGraph(), defaultdict(Counter)

###############################################################################
#                            Crawler Classes                                  #
###############################################################################

class OptimizedCrawler:
    """Memory-efficient web crawler with enhanced analytics"""
    
    def __init__(self, seed_url, max_pages=5000, max_depth=5, delay=0.1):
        self.seed_url = seed_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
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
                <div class="crawl-progress">
                <h4>🕷️ Crawling Progress</h4>
                <p><strong>Current:</strong> {current_url[:45]}...</p>
                <p><strong>Progress:</strong> {len(visited)}/{self.max_pages}</p>
                <p><strong>Depth:</strong> {depth}/{self.max_depth}</p>
                </div>
                """, unsafe_allow_html=True)
                
                stats_text.markdown(f"""
                <div class="section-stats">
                <h4>📊 Live Statistics</h4>
                <p><strong>Speed:</strong> {pages_per_second:.1f} pages/sec</p>
                <p><strong>Links:</strong> {self.crawl_stats['links_found']}</p>
                <p><strong>Queue:</strong> {len(to_visit)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                live_metrics.markdown(f"""
                <div class="metric-container">
                <h4>⚡ Performance</h4>
                <p><strong>Errors:</strong> {self.crawl_stats['errors']}</p>
                <p><strong>Time:</strong> {elapsed_time:.1f}s</p>
                <p><strong>ETA:</strong> {((self.max_pages - len(visited)) / max(pages_per_second, 0.1)):.0f}s</p>
                </div>
                """, unsafe_allow_html=True)

                # Make request with increased timeout
                response = self.session.get(current_url, timeout=15)
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
                            
                    except Exception:
                        continue
                
                # Add new URLs to queue
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
        <div class="success-box">
        <h4>✅ Crawling Complete!</h4>
        <p><strong>Pages:</strong> {len(visited)}</p>
        <p><strong>Links:</strong> {self.crawl_stats['links_found']}</p>
        <p><strong>Time:</strong> {total_time:.1f}s</p>
        </div>
        """, unsafe_allow_html=True)
        
        return visited

###############################################################################
#                            PageRank Calculation                             #
###############################################################################

def calculate_pagerank_safe(graph, alpha=0.85):
    """Calculate PageRank scores - fixed for caching issues"""
    try:
        if len(graph.edges()) > 0:
            return nx.pagerank(graph, alpha=alpha, max_iter=100)
        else:
            nodes = list(graph.nodes()) if graph.nodes else []
            return {n: 1.0 / max(len(nodes), 1) for n in nodes}
    except Exception as e:
        st.error(f"Error calculating PageRank: {e}")
        return {}

###############################################################################
#                         Section Linking Analysis                            #
###############################################################################

def analyze_section_linking_matrix(graph, section_mapping):
    """Analyze how sections link to each other with detailed metrics"""
    section_links = defaultdict(lambda: defaultdict(int))
    section_totals = defaultdict(int)
    section_pages = defaultdict(set)
    
    for source_url in graph.nodes():
        source_section = section_mapping.get(source_url, 'Other')
        section_pages[source_section].add(source_url)
        
        for target_url in graph.successors(source_url):
            target_section = section_mapping.get(target_url, 'Other')
            section_links[source_section][target_section] += 1
            section_totals[source_section] += 1
    
    # Convert to percentages with additional metrics
    linking_matrix = {}
    for source_section, targets in section_links.items():
        linking_matrix[source_section] = {}
        total_links = section_totals[source_section]
        source_page_count = len(section_pages[source_section])
        
        for target_section, link_count in targets.items():
            percentage = (link_count / total_links * 100) if total_links > 0 else 0
            avg_links_per_page = link_count / source_page_count if source_page_count > 0 else 0
            
            linking_matrix[source_section][target_section] = {
                'link_count': link_count,
                'percentage': percentage,
                'source_pages': source_page_count,
                'avg_links_per_page': avg_links_per_page,
                'link_density': (link_count / (source_page_count * len(section_pages[target_section]))) * 100 if source_page_count > 0 and len(section_pages[target_section]) > 0 else 0
            }
    
    return linking_matrix

def analyze_anchor_texts_by_section(anchor_texts, section_mapping):
    """Analyze anchor texts used when linking to each section with enhanced metrics"""
    section_anchors = defaultdict(Counter)
    section_urls = defaultdict(set)
    
    for target_url, anchors in anchor_texts.items():
        target_section = section_mapping.get(target_url, 'Other')
        section_urls[target_section].add(target_url)
        for anchor_text, count in anchors.items():
            section_anchors[target_section][anchor_text] += count
    
    # Get comprehensive anchor text analysis for each section
    section_anchor_analysis = {}
    for section, anchors in section_anchors.items():
        total_anchors = sum(anchors.values())
        unique_anchors = len(anchors)
        top_anchors = anchors.most_common(15)
        urls_count = len(section_urls[section])
        
        # Calculate anchor diversity
        anchor_diversity = unique_anchors / max(urls_count, 1)
        
        # Find keyword patterns
        keyword_patterns = {}
        for anchor, count in top_anchors:
            words = anchor.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + count
        
        top_keywords = sorted(keyword_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        section_anchor_analysis[section] = {
            'total_anchor_instances': total_anchors,
            'unique_anchor_texts': unique_anchors,
            'urls_with_anchors': urls_count,
            'anchor_diversity': anchor_diversity,
            'avg_anchors_per_url': total_anchors / max(urls_count, 1),
            'top_anchors': [
                {
                    'anchor_text': anchor,
                    'count': count,
                    'percentage': (count / total_anchors * 100) if total_anchors > 0 else 0
                }
                for anchor, count in top_anchors
            ],
            'top_keywords': [
                {
                    'keyword': keyword,
                    'frequency': freq,
                    'percentage': (freq / total_anchors * 100) if total_anchors > 0 else 0
                }
                for keyword, freq in top_keywords
            ]
        }
    
    return section_anchor_analysis

def analyze_priority_page_linking(priority_df, graph, section_mapping, anchor_texts):
    """Analyze how well priority pages are linked from different sections"""
    priority_linking_analysis = []
    
    for _, row in priority_df.iterrows():
        priority_url = str(row['url'])
        target_keyword = str(row['keyword'])
        priority_category = str(row['category'])
        
        # Get incoming links to this priority page
        incoming_links = list(graph.predecessors(priority_url)) if graph.has_node(priority_url) else []
        
        # Analyze by source section
        section_links = defaultdict(list)
        for source_url in incoming_links:
            source_section = section_mapping.get(source_url, 'Other')
            section_links[source_section].append(source_url)
        
        # Analyze anchor texts for this priority page
        priority_anchors = anchor_texts.get(priority_url, Counter())
        total_anchors = sum(priority_anchors.values())
        
        # Check keyword usage in anchors
        keyword_usage = 0
        keyword_variants = []
        if target_keyword and target_keyword != 'nan':
            for anchor, count in priority_anchors.items():
                if target_keyword.lower() in anchor.lower():
                    keyword_usage += count
                    keyword_variants.append({'anchor': anchor, 'count': count})
        
        keyword_percentage = (keyword_usage / total_anchors * 100) if total_anchors > 0 else 0
        
        priority_linking_analysis.append({
            'url': priority_url,
            'category': priority_category,
            'target_keyword': target_keyword,
            'total_incoming_links': len(incoming_links),
            'linking_sections': dict(section_links),
            'total_anchor_instances': total_anchors,
            'keyword_anchor_usage': keyword_usage,
            'keyword_anchor_percentage': keyword_percentage,
            'keyword_variants': keyword_variants,
            'top_anchors': [
                {'anchor': anchor, 'count': count, 'percentage': (count/total_anchors*100)}
                for anchor, count in priority_anchors.most_common(10)
            ] if total_anchors > 0 else []
        })
    
    return priority_linking_analysis

###############################################################################
#                              OpenAI Integration                             #
###############################################################################

def get_ai_recommendations(section_analysis, top_pages, priority_analysis, linking_matrix, anchor_analysis, openai_api_key):
    """Get comprehensive AI-powered recommendations using OpenAI"""
    if not openai_api_key or len(openai_api_key.strip()) < 10:
        return "No OpenAI API key provided."
    
    try:
        client = OpenAI(api_key=openai_api_key.strip())
        
        # Prepare comprehensive data summary for AI
        data_summary = {
            "top_sections": section_analysis[:5],
            "top_pages": top_pages[:10],
            "priority_status": priority_analysis[:5],
            "linking_patterns": {
                section: {
                    target: data['percentage'] 
                    for target, data in targets.items() 
                    if data['percentage'] > 5
                }[:3]  # Top 3 targets per section
                for section, targets in linking_matrix.items()
            },
            "anchor_insights": {
                section: {
                    'diversity': data['anchor_diversity'],
                    'top_keywords': [kw['keyword'] for kw in data['top_keywords'][:5]]
                }
                for section, data in anchor_analysis.items()
            }
        }
        
        prompt = f"""
        As a senior SEO consultant, analyze this comprehensive PageRank distribution data:
        
        TOP SECTIONS BY PAGERANK:
        {data_summary['top_sections']}
        
        TOP PAGES BY PAGERANK:
        {data_summary['top_pages']}
        
        PRIORITY PAGES STATUS:
        {data_summary['priority_status']}
        
        INTERNAL LINKING PATTERNS:
        {data_summary['linking_patterns']}
        
        ANCHOR TEXT INSIGHTS:
        {data_summary['anchor_insights']}
        
        Provide specific, actionable recommendations for:
        
        1. INTERNAL LINKING STRATEGIES
           - How to boost PageRank flow to underperforming priority pages
           - Which sections should link more to priority pages
           - Optimal linking ratios between sections
        
        2. PAGERANK WASTE REDUCTION
           - Which sections are receiving too much PageRank
           - How to redirect PageRank from low-value to high-value sections
           - Specific pages to reduce internal links to
        
        3. ANCHOR TEXT OPTIMIZATION
           - Keyword opportunities in anchor texts
           - Sections with poor anchor text diversity
           - Target keywords that need more anchor text usage
        
        4. SECTION-TO-SECTION LINKING IMPROVEMENTS
           - Underutilized linking opportunities between sections
           - Sections that should increase their outbound linking
           - Cross-section linking strategies for better PageRank distribution
        
        5. PRIORITY PAGE OPTIMIZATION
           - Specific recommendations for each underperforming priority page
           - Quick wins for improving priority page PageRank
           - Long-term strategies for priority page authority building
        
        Format as numbered sections with specific, implementable recommendations.
        Include metrics and percentage improvements where possible.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a senior SEO consultant specializing in internal linking, PageRank optimization, and technical SEO. Provide specific, actionable recommendations with measurable outcomes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error getting AI recommendations: {str(e)}"

###############################################################################
#                             Advanced Visualizations                         #
###############################################################################

def create_section_heatmap_with_anchors(linking_matrix, anchor_analysis):
    """Create an advanced heatmap showing section-to-section linking with anchor text insights"""
    
    if not linking_matrix:
        return go.Figure()
    
    # Prepare data for heatmap
    sections = list(set(list(linking_matrix.keys()) + [target for targets in linking_matrix.values() for target in targets.keys()]))
    
    # Create matrix data
    matrix_data = []
    section_labels = []
    
    for i, source in enumerate(sections):
        row_data = []
        for j, target in enumerate(sections):
            if source in linking_matrix and target in linking_matrix[source]:
                percentage = linking_matrix[source][target]['percentage']
                link_count = linking_matrix[source][target]['link_count']
            else:
                percentage = 0
                link_count = 0
            row_data.append(percentage)
        matrix_data.append(row_data)
        section_labels.append(f"{source}<br>({anchor_analysis.get(source, {}).get('unique_anchor_texts', 0)} anchors)")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=[f"{s}<br>({anchor_analysis.get(s, {}).get('total_anchor_instances', 0)})" for s in sections],
        y=section_labels,
        colorscale='RdYlBu_r',
        text=[[f"{val:.1f}%<br>{linking_matrix.get(sections[i], {}).get(sections[j], {}).get('link_count', 0)} links" 
               if val > 0 else "" for j, val in enumerate(row)] for i, row in enumerate(matrix_data)],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="Section-to-Section Linking Heatmap with Anchor Text Metrics",
        xaxis_title="Target Sections (Total Anchor Instances)",
        yaxis_title="Source Sections (Unique Anchor Texts)",
        height=600,
        width=800
    )
    
    return fig

def create_priority_page_performance_chart(priority_linking_analysis, pagerank_scores):
    """Create comprehensive chart showing priority page performance"""
    
    if not priority_linking_analysis:
        return go.Figure()
    
    # Prepare data
    urls = []
    pageranks = []
    incoming_links = []
    keyword_usage = []
    categories = []
    
    for analysis in priority_linking_analysis:
        url = analysis['url']
        urls.append(url.split('/')[-1] or 'Homepage')  # Get page name
        pageranks.append(pagerank_scores.get(url, 0))
        incoming_links.append(analysis['total_incoming_links'])
        keyword_usage.append(analysis['keyword_anchor_percentage'])
        categories.append(analysis['category'])
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PageRank Scores', 'Incoming Links', 'Keyword Anchor Usage (%)', 'Performance Matrix'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "scatter"}]]
    )
    
    # PageRank bar chart
    fig.add_trace(
        go.Bar(x=urls, y=pageranks, name="PageRank", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Incoming links bar chart
    fig.add_trace(
        go.Bar(x=urls, y=incoming_links, name="Incoming Links", marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Keyword usage bar chart
    fig.add_trace(
        go.Bar(x=urls, y=keyword_usage, name="Keyword Usage %", marker_color='orange'),
        row=2, col=1
    )
    
    # Performance matrix scatter plot
    fig.add_trace(
        go.Scatter(
            x=pageranks,
            y=keyword_usage,
            mode='markers+text',
            text=urls,
            textposition='top center',
            marker=dict(
                size=[x*100 for x in pageranks],  # Size based on PageRank
                color=incoming_links,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Incoming Links")
            ),
            name="Performance Matrix"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Priority Pages Comprehensive Performance Analysis",
        showlegend=False
    )
    
    return fig

def create_anchor_text_word_cloud_chart(anchor_analysis):
    """Create word frequency chart from anchor text analysis"""
    
    if not anchor_analysis:
        return go.Figure()
    
    # Aggregate all keywords across sections
    all_keywords = []
    keyword_weights = []
    sections = []
    
    for section, data in anchor_analysis.items():
        for keyword_data in data.get('top_keywords', [])[:10]:
            all_keywords.append(keyword_data['keyword'])
            keyword_weights.append(keyword_data['frequency'])
            sections.append(section)
    
    if not all_keywords:
        return go.Figure()
    
    # Create bubble chart
    fig = go.Figure(data=go.Scatter(
        x=list(range(len(all_keywords))),
        y=keyword_weights,
        mode='markers+text',
        text=all_keywords,
        textposition='middle center',
        marker=dict(
            size=[min(w*2, 100) for w in keyword_weights],  # Scale bubble size
            color=keyword_weights,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Frequency")
        ),
        hovertemplate='<b>%{text}</b><br>Frequency: %{y}<br>Section: %{customdata}<extra></extra>',
        customdata=sections
    ))
    
    fig.update_layout(
        title="Top Anchor Text Keywords Across All Sections",
        xaxis_title="Keywords",
        yaxis_title="Frequency",
        height=500,
        xaxis=dict(showticklabels=False)
    )
    
    return fig

###############################################################################
#                             Tab Functions                                   #
###############################################################################

def tab_section_analysis(pagerank_scores, section_mapping, anchor_analysis, section_matrix, openai_key):
    """Enhanced section analysis tab with comprehensive insights"""
    st.header("📊 Section-by-Section PageRank Analysis")
    
    # Calculate section PageRank distribution
    section_pr = defaultdict(float)
    section_counts = defaultdict(int)
    
    for url, score in pagerank_scores.items():
        section = section_mapping.get(url, 'Other')
        section_pr[section] += score
        section_counts[section] += 1
    
    total_pr = sum(section_pr.values())
    
    # Create comprehensive section analysis
    section_analysis = []
    for section, pr_score in section_pr.items():
        percentage = (pr_score / total_pr * 100) if total_pr > 0 else 0
        avg_pr = pr_score / section_counts[section] if section_counts[section] > 0 else 0
        
        # Get anchor text metrics for this section
        anchor_metrics = anchor_analysis.get(section, {})
        
        section_analysis.append({
            'Section': section,
            'Total PageRank': pr_score,
            'Percentage': percentage,
            'Page Count': section_counts[section],
            'Avg PageRank': avg_pr,
            'Unique Anchors': anchor_metrics.get('unique_anchor_texts', 0),
            'Anchor Diversity': anchor_metrics.get('anchor_diversity', 0),
            'Total Anchor Instances': anchor_metrics.get('total_anchor_instances', 0)
        })
    
    # Sort by total PageRank
    section_analysis.sort(key=lambda x: x['Total PageRank'], reverse=True)
    section_df = pd.DataFrame(section_analysis)
    section_df = fix_dataframe_types(section_df)
    
    # Display top 3 sections with safe access - FIXED ERROR
    if len(section_df) >= 3:
        top_3 = section_df.head(3).reset_index(drop=True)
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Top 3 Sections by PageRank</h4>
        <ul>
        <li><strong>🥇 {safe_iloc_access(top_3, 0, 'Section')}:</strong> {float(safe_iloc_access(top_3, 0, 'Percentage')):.1f}% ({int(safe_iloc_access(top_3, 0, 'Page Count'))} pages)</li>
        <li><strong>🥈 {safe_iloc_access(top_3, 1, 'Section')}:</strong> {float(safe_iloc_access(top_3, 1, 'Percentage')):.1f}% ({int(safe_iloc_access(top_3, 1, 'Page Count'))} pages)</li>
        <li><strong>🥉 {safe_iloc_access(top_3, 2, 'Section')}:</strong> {float(safe_iloc_access(top_3, 2, 'Percentage')):.1f}% ({int(safe_iloc_access(top_3, 2, 'Page Count'))} pages)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display comprehensive section DataFrame
    st.subheader("📋 Detailed Section Analysis")
    st.dataframe(section_df, use_container_width=True)
    
    # Create enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # PageRank distribution chart
        fig_pr = px.bar(
            section_df,
            x='Section',
            y='Total PageRank',
            title='PageRank Distribution by Section',
            color='Percentage',
            color_continuous_scale='Blues'
        )
        fig_pr.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_pr, use_container_width=True)
    
    with col2:
        # Anchor diversity vs PageRank scatter
        fig_diversity = px.scatter(
            section_df,
            x='Anchor Diversity',
            y='Total PageRank',
            size='Page Count',
            color='Percentage',
            title='Anchor Diversity vs PageRank',
            hover_data=['Section', 'Unique Anchors']
        )
        fig_diversity.update_layout(height=500)
        st.plotly_chart(fig_diversity, use_container_width=True)
    
    # Section linking heatmap
    if section_matrix:
        st.subheader("🔗 Section-to-Section Linking Heatmap")
        heatmap_fig = create_section_heatmap_with_anchors(section_matrix, anchor_analysis)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Detailed linking matrix analysis
        st.subheader("📊 Detailed Linking Analysis")
        
        # Convert linking matrix to DataFrame for better display
        linking_data = []
        for source, targets in section_matrix.items():
            for target, data in targets.items():
                if data['percentage'] > 1:  # Only show significant links
                    linking_data.append({
                        'Source Section': source,
                        'Target Section': target,
                        'Link Count': data['link_count'],
                        'Percentage': data['percentage'],
                        'Source Pages': data['source_pages'],
                        'Avg Links/Page': data['avg_links_per_page'],
                        'Link Density': data['link_density']
                    })
        
        if linking_data:
            linking_df = pd.DataFrame(linking_data)
            linking_df = fix_dataframe_types(linking_df)
            st.dataframe(linking_df.sort_values('Percentage', ascending=False), use_container_width=True)
    
    # Anchor text analysis by section
    st.subheader("🏷️ Anchor Text Analysis by Section")
    
    for section, data in list(anchor_analysis.items())[:5]:  # Show top 5 sections
        with st.expander(f"{section} - {data['total_anchor_instances']} total anchors, {data['unique_anchor_texts']} unique"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Anchor Texts:**")
                if data['top_anchors']:
                    anchor_df = pd.DataFrame(data['top_anchors'][:10])
                    anchor_df = fix_dataframe_types(anchor_df)
                    st.dataframe(anchor_df, use_container_width=True)
            
            with col2:
                st.markdown("**Top Keywords:**")
                if data['top_keywords']:
                    keyword_df = pd.DataFrame(data['top_keywords'][:10])
                    keyword_df = fix_dataframe_types(keyword_df)
                    st.dataframe(keyword_df, use_container_width=True)
            
            # Metrics
            st.markdown(f"""
            <div class="section-stats">
            <p><strong>Anchor Diversity:</strong> {data['anchor_diversity']:.2f}</p>
            <p><strong>Avg Anchors per URL:</strong> {data['avg_anchors_per_url']:.1f}</p>
            <p><strong>URLs with Anchors:</strong> {data['urls_with_anchors']}</p>
            </div>
            """, unsafe_allow_html=True)

def tab_top_pages_analysis(pagerank_scores, section_mapping, page_data):
    """Enhanced top pages analysis with comprehensive metrics"""
    st.header("🏆 Top Pages Analysis")
    
    # Sort pages by PageRank (highest to lowest) - ADDRESSES FEEDBACK
    sorted_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create comprehensive top pages analysis
    top_pages_data = []
    for i, (url, score) in enumerate(sorted_pages[:50], 1):  # Top 50 pages
        section = section_mapping.get(url, 'Other')
        page_info = page_data.get(url, {})
        
        # Create readable page name
        parsed = urlparse(url)
        page_name = parsed.path.split('/')[-1] or 'Homepage'
        if not page_name or page_name in ['index.html', 'index.php']:
            page_name = 'Homepage'
        
        top_pages_data.append({
            'Rank': i,
            'Page': page_name[:40] + '...' if len(page_name) > 40 else page_name,
            'URL': url,
            'PageRank': score,
            'Section': section,
            'Word Count': page_info.get('word_count', 0),
            'Internal Links': page_info.get('internal_links', 0),
            'Title': page_info.get('title', '')[:50] + '...' if len(page_info.get('title', '')) > 50 else page_info.get('title', ''),
            'Route Depth': page_info.get('route_depth', 0)
        })
    
    top_pages_df = pd.DataFrame(top_pages_data)
    top_pages_df = fix_dataframe_types(top_pages_df)
    
    # Display comprehensive metrics for top 20
    if len(top_pages_df) >= 20:
        top_20 = top_pages_df.head(20).reset_index(drop=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Highest PageRank", f"{float(safe_iloc_access(top_20, 0, 'PageRank')):.6f}")
        with col2:
            st.metric("Average PageRank", f"{top_20['PageRank'].astype(float).mean():.6f}")
        with col3:
            most_common_section = top_20['Section'].mode()
            if not most_common_section.empty:
                st.metric("Most Common Section", most_common_section.iloc[0])
        with col4:
            st.metric("Avg Internal Links", f"{top_20['Internal Links'].astype(int).mean():.0f}")
    
    # Display sortable table with enhanced data
    st.subheader("📋 Top Pages Details (Sorted by PageRank)")
    st.dataframe(top_pages_df, use_container_width=True)
    
    # Create comprehensive visualizations
    if len(top_pages_df) >= 20:
        top_20_viz = top_pages_df.head(20).reset_index(drop=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PageRank distribution by rank
            fig_rank = px.bar(
                top_20_viz,
                x='Rank',
                y='PageRank',
                color='Section',
                title='Top 20 Pages by PageRank Score',
                hover_data=['Page', 'Word Count', 'Internal Links']
            )
            fig_rank.update_layout(height=500)
            st.plotly_chart(fig_rank, use_container_width=True)
        
        with col2:
            # PageRank vs Internal Links correlation
            fig_corr = px.scatter(
                top_20_viz,
                x='Internal Links',
                y='PageRank',
                color='Section',
                size='Word Count',
                title='PageRank vs Internal Links',
                hover_data=['Page', 'Rank']
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Section-wise top pages analysis
    st.subheader("🎯 Top Pages by Section")
    
    section_top_pages = defaultdict(list)
    for _, row in top_pages_df.iterrows():
        section_top_pages[row['Section']].append(row)
    
    for section, pages in list(section_top_pages.items())[:5]:  # Top 5 sections
        with st.expander(f"{section} - Top Pages"):
            section_df = pd.DataFrame(pages[:10])  # Top 10 per section
            section_df = fix_dataframe_types(section_df)
            st.dataframe(section_df[['Rank', 'Page', 'PageRank', 'Word Count', 'Internal Links']], use_container_width=True)

def tab_priority_pages_analysis(pagerank_scores, section_mapping, priority_pages_df, graph, anchor_texts):
    """Comprehensive priority pages analysis with linking insights"""
    st.header("🎯 Priority Pages Analysis")
    
    if priority_pages_df is None or priority_pages_df.empty:
        st.warning("No priority pages data available. Please upload a priority pages CSV.")
        return
    
    # Comprehensive priority page analysis
    priority_analysis = analyze_priority_page_linking(priority_pages_df, graph, section_mapping, anchor_texts)
    
    # Calculate summary statistics
    total_priority = len(priority_analysis)
    found_in_crawl = sum(1 for p in priority_analysis if p['url'] in pagerank_scores)
    top_20_urls = set([url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]])
    in_top_20_count = sum(1 for p in priority_analysis if p['url'] in top_20_urls)
    
    # Calculate average metrics
    avg_pagerank = np.mean([pagerank_scores.get(p['url'], 0) for p in priority_analysis])
    avg_incoming_links = np.mean([p['total_incoming_links'] for p in priority_analysis])
    avg_keyword_usage = np.mean([p['keyword_anchor_percentage'] for p in priority_analysis])
    
    # Display comprehensive metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📊 Coverage</h4>
        <p><strong>Total:</strong> {total_priority}</p>
        <p><strong>Found:</strong> {found_in_crawl} ({(found_in_crawl/total_priority*100):.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
        <h4>🏆 Performance</h4>
        <p><strong>In Top 20:</strong> {in_top_20_count}</p>
        <p><strong>Success Rate:</strong> {(in_top_20_count/total_priority*100):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
        <h4>📈 Averages</h4>
        <p><strong>PageRank:</strong> {avg_pagerank:.6f}</p>
        <p><strong>Links:</strong> {avg_incoming_links:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
        <h4>🏷️ Keywords</h4>
        <p><strong>Avg Usage:</strong> {avg_keyword_usage:.1f}%</p>
        <p><strong>Optimization:</strong> {'Good' if avg_keyword_usage > 20 else 'Needs Work'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create comprehensive priority pages DataFrame
    priority_detailed_data = []
    for analysis in priority_analysis:
        url = analysis['url']
        priority_detailed_data.append({
            'URL': url,
            'Category': analysis['category'],
            'Target Keyword': analysis['target_keyword'],
            'PageRank': pagerank_scores.get(url, 0.0),
            'Rank Position': next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == url), len(pagerank_scores)),
            'In Top 20': url in top_20_urls,
            'Incoming Links': analysis['total_incoming_links'],
            'Section': section_mapping.get(url, 'Not Found'),
            'Keyword Usage %': analysis['keyword_anchor_percentage'],
            'Total Anchors': analysis['total_anchor_instances']
        })
    
    priority_detailed_df = pd.DataFrame(priority_detailed_data)
    priority_detailed_df = fix_dataframe_types(priority_detailed_df)
    
    # Display detailed priority pages analysis
    st.subheader("📋 Detailed Priority Pages Analysis")
    st.dataframe(priority_detailed_df, use_container_width=True)
    
    # Create priority page performance visualization
    st.subheader("📊 Priority Page Performance Visualization")
    priority_perf_chart = create_priority_page_performance_chart(priority_analysis, pagerank_scores)
    st.plotly_chart(priority_perf_chart, use_container_width=True)
    
    # Identify improvement opportunities
    st.subheader("⚡ Improvement Opportunities")
    
    # Low performing priority pages
    low_performing = [p for p in priority_analysis if pagerank_scores.get(p['url'], 0) > 0 and 
                     next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == p['url']), len(pagerank_scores)) > 50]
    
    if low_performing:
        st.markdown("**🔍 Pages Needing Attention:**")
        for i, page in enumerate(low_performing[:5], 1):
            url = page['url']
            pagerank = pagerank_scores.get(url, 0)
            rank_position = next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == url), len(pagerank_scores))
            
            # Find linking opportunities
            top_pages_not_linking = []
            for top_url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]:
                if not graph.has_edge(top_url, url) and top_url != url:
                    top_pages_not_linking.append(top_url)
            
            st.markdown(f"""
            <div class="low-performance">
            <h5>{i}. {url}</h5>
            <p><strong>Current Rank:</strong> #{rank_position} | <strong>PageRank:</strong> {pagerank:.6f}</p>
            <p><strong>Incoming Links:</strong> {page['total_incoming_links']} | <strong>Keyword Usage:</strong> {page['keyword_anchor_percentage']:.1f}%</p>
            <p><strong>Quick Win:</strong> Add internal links from {len(top_pages_not_linking)} high-authority pages</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Keyword optimization opportunities
    st.subheader("🏷️ Keyword Optimization Opportunities")
    
    keyword_opportunities = [p for p in priority_analysis if p['target_keyword'] != 'nan' and p['keyword_anchor_percentage'] < 15 and p['total_anchor_instances'] > 5]
    
    if keyword_opportunities:
        for opp in keyword_opportunities[:5]:
            st.markdown(f"""
            <div class="linking-opportunity">
            <h5>🎯 {opp['url']}</h5>
            <p><strong>Target Keyword:</strong> "{opp['target_keyword']}"</p>
            <p><strong>Current Usage:</strong> {opp['keyword_anchor_percentage']:.1f}% ({opp['keyword_anchor_usage']} out of {opp['total_anchor_instances']} anchors)</p>
            <p><strong>Opportunity:</strong> Increase keyword usage in anchor texts by {20 - opp['keyword_anchor_percentage']:.1f} percentage points</p>
            </div>
            """, unsafe_allow_html=True)

def tab_five_critical_questions(pagerank_scores, section_mapping, priority_pages_df, anchor_analysis, graph):
    """Comprehensive analysis of the five critical PageRank distribution questions - FIXED ERRORS"""
    st.markdown("""
    <div class="five-questions-header">
    <h2>🔍 Five Critical PageRank Distribution Questions</h2>
    <p>Comprehensive analysis addressing the key questions for PageRank optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Question 1: Which Sections receive the most PR?
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.subheader("1️⃣ Which Sections of the site are receiving the most PR?")
    
    section_pr = defaultdict(float)
    for url, score in pagerank_scores.items():
        section = section_mapping.get(url, 'Other')
        section_pr[section] += score
    
    total_pr = sum(section_pr.values())
    sorted_sections = sorted(section_pr.items(), key=lambda x: x[1], reverse=True)
    
    # minimal fix using explicit indexing
    if len(sorted_sections) >= 3 and total_pr > 0:
        st.markdown(f"""
**Top 3 Sections:**
1. **{sorted_sections[0][0]}**: {(sorted_sections[0][1]/total_pr*100):.1f}%
2. **{sorted_sections[1][0]}**: {(sorted_sections[1][1]/total_pr*100):.1f}%
3. **{sorted_sections[2][0]}**: {(sorted_sections[2][1]/total_pr*100):.1f}%
""")

        
        # Create visualization
        top_5_sections = sorted_sections[:5]
        section_chart_df = pd.DataFrame([
            {'Section': section, 'PageRank': pr, 'Percentage': (pr/total_pr*100)}
            for section, pr in top_5_sections
        ])
        section_chart_df = fix_dataframe_types(section_chart_df)
        
        fig_sections = px.pie(
            section_chart_df,
            values='PageRank',
            names='Section',
            title='PageRank Distribution - Top 5 Sections'
        )
        st.plotly_chart(fig_sections, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 2: Which specific pages receive the most PR?
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.subheader("2️⃣ Which specific pages receive the most PR?")
    
    top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    st.markdown("**Top 10 Pages by PageRank:**")
    for i, (url, score) in enumerate(top_pages, 1):
        section = section_mapping.get(url, 'Other')
        page_name = urlparse(url).path or '/'
        st.markdown(f"{i}. **{page_name}** ({section}) - {score:.6f}")
    
    # Create top pages visualization
    top_pages_chart_df = pd.DataFrame([
        {
            'Rank': i,
            'Page': urlparse(url).path.split('/')[-1] or 'Homepage',
            'PageRank': score,
            'Section': section_mapping.get(url, 'Other')
        }
        for i, (url, score) in enumerate(top_pages, 1)
    ])
    top_pages_chart_df = fix_dataframe_types(top_pages_chart_df)
    
    fig_pages = px.bar(
        top_pages_chart_df,
        x='Rank',
        y='PageRank',
        color='Section',
        title='Top 10 Pages by PageRank',
        hover_data=['Page']
    )
    st.plotly_chart(fig_pages, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 3: Do these align with Priority Target Pages?
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.subheader("3️⃣ Do these align with your Priority Target Pages?")
    
    if priority_pages_df is not None and not priority_pages_df.empty:
        priority_urls = set(priority_pages_df['url'].astype(str))
        top_20_urls = set([url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]])
        
        alignment = len(priority_urls.intersection(top_20_urls))
        total_priority = len(priority_urls)
        alignment_percentage = (alignment/total_priority*100) if total_priority > 0 else 0
        
        st.markdown(f"""
        **Alignment Analysis:**
        - Priority pages in top 20: {alignment}/{total_priority} ({alignment_percentage:.1f}%)
        - Alignment status: {'✅ Good' if alignment_percentage > 50 else '⚠️ Needs Improvement' if alignment_percentage > 25 else '❌ Poor'}
        """)
        
        # Show specific alignments and misalignments
        aligned_pages = priority_urls.intersection(top_20_urls)
        misaligned_pages = priority_urls - top_20_urls
        
        if aligned_pages:
            st.markdown("**✅ Well-performing priority pages:**")
            for page in list(aligned_pages)[:5]:
                rank = next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == page), None)
                st.markdown(f"- {page} (Rank #{rank})")
        
        if misaligned_pages:
            st.markdown("**⚠️ Underperforming priority pages:**")
            for page in list(misaligned_pages)[:5]:
                rank = next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == page), 'Not found')
                st.markdown(f"- {page} (Rank #{rank})")
    else:
        st.warning("No priority pages data available for alignment analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 4: How to reduce PR to non-valuable sections?
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.subheader("4️⃣ How can we reduce the PR to non-valuable sections?")
    
    # Identify potentially low-value sections
    low_value_keywords = ['tag', 'archive', 'category', 'search', 'pagination']
    low_value_sections = []
    
    for section, pr_amount in sorted_sections:
        if any(keyword in section.lower() for keyword in low_value_keywords):
            percentage = (pr_amount / total_pr * 100)
            if percentage > 5:  # Only flag if > 5% of total PR
                low_value_sections.append((section, percentage, pr_amount))
    
    if low_value_sections:
        st.markdown("**Sections with potentially wasted PR:**")
        total_waste = 0
        for section, percentage, pr_amount in low_value_sections:
            total_waste += pr_amount
            st.markdown(f"- **{section}**: {percentage:.1f}% of total PR ({pr_amount:.6f})")
        
        st.markdown(f"""
        <div class="warning-box">
        <p><strong>Total PR potentially wasted:</strong> {total_waste:.6f} ({(total_waste/total_pr*100):.1f}%)</p>
        <p><strong>Recommendation:</strong> Reduce internal links to these sections and redirect link equity to priority pages.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
        <p>✅ No significant PR waste detected in obvious low-value sections</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 5: How to redirect PR to priority pages?
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    st.subheader("5️⃣ How can we redirect this PR to priority Pages?")
    
    if priority_pages_df is not None and not priority_pages_df.empty:
        st.markdown("**Top Linking Opportunities:**")
        
        # Find linking opportunities for each priority page
        high_pr_pages = [url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]]
        
        opportunity_count = 0
        for _, row in priority_pages_df.iterrows():
            priority_url = str(row['url'])
            if priority_url in pagerank_scores:
                priority_pr = pagerank_scores[priority_url]
                
                # Find high-PR pages that don't link to this priority page
                not_linking = []
                for high_pr_url in high_pr_pages[:10]:  # Check top 10
                    if high_pr_url != priority_url and not graph.has_edge(high_pr_url, priority_url):
                        high_pr_score = pagerank_scores[high_pr_url]
                        if high_pr_score > priority_pr * 1.5:  # Only suggest if significantly higher PR
                            not_linking.append({
                                'source': high_pr_url,
                                'source_section': section_mapping.get(high_pr_url, 'Other'),
                                'source_pr': high_pr_score,
                                'potential_boost': high_pr_score * 0.15  # Estimated 15% PR transfer
                            })
                
                if not_linking and opportunity_count < 5:  # Show top 5 opportunities
                    opportunity_count += 1
                    best_opportunity = max(not_linking, key=lambda x: x['potential_boost'])
                    
                    st.markdown(f"""
                    <div class="linking-opportunity">
                    <h5>🎯 Opportunity #{opportunity_count}: {priority_url}</h5>
                    <p><strong>Current PageRank:</strong> {priority_pr:.6f}</p>
                    <p><strong>Best Link Source:</strong> {best_opportunity['source']}</p>
                    <p><strong>Source Section:</strong> {best_opportunity['source_section']}</p>
                    <p><strong>Source PageRank:</strong> {best_opportunity['source_pr']:.6f}</p>
                    <p><strong>Estimated Boost:</strong> +{best_opportunity['potential_boost']:.6f} ({(best_opportunity['potential_boost']/priority_pr*100):.1f}% increase)</p>
                    <p><strong>Action:</strong> Add contextual internal link with relevant anchor text</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        if opportunity_count == 0:
            st.info("Most priority pages are already well-linked from high-authority pages.")
    else:
        st.warning("Upload priority pages CSV to see specific linking opportunities")
    
    st.markdown('</div>', unsafe_allow_html=True)

def tab_anchor_text_analysis(anchor_texts, section_mapping, priority_pages_df):
    """Comprehensive anchor text analysis with keyword insights"""
    st.header("🔗 Comprehensive Anchor Text Analysis")
    
    # Analyze anchor texts by section
    section_anchors = analyze_anchor_texts_by_section(anchor_texts, section_mapping)
    
    # Create anchor text word cloud visualization
    st.subheader("☁️ Anchor Text Keyword Frequency")
    wordcloud_chart = create_anchor_text_word_cloud_chart(section_anchors)
    st.plotly_chart(wordcloud_chart, use_container_width=True)
    
    # Section-based anchor text analysis
    st.subheader("📊 Anchor Text Analysis by Section")
    
    # Create summary DataFrame
    anchor_summary_data = []
    for section, data in section_anchors.items():
        anchor_summary_data.append({
            'Section': section,
            'Total Anchor Instances': data['total_anchor_instances'],
            'Unique Anchor Texts': data['unique_anchor_texts'],
            'URLs with Anchors': data['urls_with_anchors'],
            'Anchor Diversity': data['anchor_diversity'],
            'Avg Anchors per URL': data['avg_anchors_per_url']
        })
    
    anchor_summary_df = pd.DataFrame(anchor_summary_data)
    anchor_summary_df = fix_dataframe_types(anchor_summary_df)
    st.dataframe(anchor_summary_df.sort_values('Total Anchor Instances', ascending=False), use_container_width=True)
    
    # Detailed section analysis
    for section, data in list(section_anchors.items())[:8]:  # Show top 8 sections
        with st.expander(f"{section} - {data['total_anchor_instances']} total anchors"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Anchor Texts:**")
                if data['top_anchors']:
                    anchor_df = pd.DataFrame(data['top_anchors'][:15])
                    anchor_df = fix_dataframe_types(anchor_df)
                    
                    # Create bar chart for top anchors
                    fig_anchors = px.bar(
                        anchor_df,
                        x='percentage',
                        y='anchor_text',
                        orientation='h',
                        title='Top Anchor Texts by Percentage',
                        color='count',
                        color_continuous_scale='Blues'
                    )
                    fig_anchors.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_anchors, use_container_width=True)
                    
                    st.dataframe(anchor_df, use_container_width=True)
            
            with col2:
                st.markdown("**Top Keywords:**")
                if data['top_keywords']:
                    keyword_df = pd.DataFrame(data['top_keywords'][:15])
                    keyword_df = fix_dataframe_types(keyword_df)
                    
                    # Create bar chart for keywords
                    fig_keywords = px.bar(
                        keyword_df,
                        x='frequency',
                        y='keyword',
                        orientation='h',
                        title='Most Frequent Keywords',
                        color='percentage',
                        color_continuous_scale='Greens'
                    )
                    fig_keywords.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_keywords, use_container_width=True)
                    
                    st.dataframe(keyword_df, use_container_width=True)
            
            # Anchor text insights
            st.markdown(f"""
            <div class="section-stats">
            <h5>📈 Section Insights</h5>
            <p><strong>Anchor Diversity Score:</strong> {data['anchor_diversity']:.2f}/10 {'(Excellent)' if data['anchor_diversity'] > 3 else '(Good)' if data['anchor_diversity'] > 1.5 else '(Needs Improvement)'}</p>
            <p><strong>Average Anchors per URL:</strong> {data['avg_anchors_per_url']:.1f}</p>
            <p><strong>Coverage:</strong> {data['urls_with_anchors']} URLs have anchor text data</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Priority pages anchor text analysis
    if priority_pages_df is not None and not priority_pages_df.empty:
        st.subheader("🎯 Priority Pages Keyword Coverage Analysis")
        
        # Analyze keyword usage for priority pages
        priority_anchor_analysis = []
        for _, row in priority_pages_df.iterrows():
            url = str(row['url'])
            target_keyword = str(row['keyword'])
            category = str(row['category'])
            
            if url in anchor_texts and target_keyword and target_keyword != 'nan':
                anchors = anchor_texts[url]
                total_anchors = sum(anchors.values())
                
                # Check various forms of keyword usage
                exact_match = sum(count for anchor, count in anchors.items() 
                                if target_keyword.lower() == anchor.lower().strip())
                
                partial_match = sum(count for anchor, count in anchors.items() 
                                  if target_keyword.lower() in anchor.lower())
                
                keyword_words = target_keyword.lower().split()
                word_matches = sum(count for anchor, count in anchors.items() 
                                 if any(word in anchor.lower() for word in keyword_words))
                
                priority_anchor_analysis.append({
                    'URL': url,
                    'Category': category,
                    'Target Keyword': target_keyword,
                    'Total Anchors': total_anchors,
                    'Exact Matches': exact_match,
                    'Partial Matches': partial_match,
                    'Word Matches': word_matches,
                    'Exact Match %': (exact_match / total_anchors * 100) if total_anchors > 0 else 0,
                    'Partial Match %': (partial_match / total_anchors * 100) if total_anchors > 0 else 0,
                    'Word Match %': (word_matches / total_anchors * 100) if total_anchors > 0 else 0,
                    'Optimization Score': ((exact_match * 3 + partial_match * 2 + word_matches) / max(total_anchors, 1) * 100)
                })
        
        if priority_anchor_analysis:
            priority_anchor_df = pd.DataFrame(priority_anchor_analysis)
            priority_anchor_df = fix_dataframe_types(priority_anchor_df)
            
            st.dataframe(priority_anchor_df.sort_values('Optimization Score', ascending=False), use_container_width=True)
            
            # Highlight optimization opportunities
            low_optimization = [item for item in priority_anchor_analysis 
                              if item['Optimization Score'] < 20 and item['Total Anchors'] > 10]
            
            if low_optimization:
                st.subheader("⚡ Anchor Text Optimization Opportunities")
                for i, item in enumerate(low_optimization[:5], 1):
                    st.markdown(f"""
                    <div class="linking-opportunity">
                    <h5>{i}. {item['URL']}</h5>
                    <p><strong>Target:</strong> "{item['Target Keyword']}" | <strong>Score:</strong> {item['Optimization Score']:.1f}/100</p>
                    <p><strong>Current Usage:</strong> {item['Exact Matches']} exact, {item['Partial Matches']} partial out of {item['Total Anchors']} total</p>
                    <p><strong>Opportunity:</strong> Increase keyword usage by {20 - item['Partial Match %']:.1f} percentage points</p>
                    <p><strong>Action:</strong> Update existing anchor texts to include target keyword variations</p>
                    </div>
                    """, unsafe_allow_html=True)

def tab_ai_insights(prio_df, pagerank_scores, section_mapping, anchor_texts, graph, openai_key):
    """Comprehensive AI-powered insights and recommendations"""
    st.header("🤖 AI-Powered SEO Insights & Recommendations")
    
    if not openai_key or len(openai_key.strip()) < 10:
        st.markdown("""
        <div class="ai-insight">
        <h4>🔑 API Key Required</h4>
        <p>Enter your OpenAI API key in the sidebar to unlock comprehensive AI-powered recommendations.</p>
        <p>Get your API key at: <a href="https://platform.openai.com" target="_blank">platform.openai.com</a></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Prepare comprehensive data for AI analysis
    st.subheader("📊 Analyzing Your Data...")
    
    with st.spinner("🧠 AI is analyzing your PageRank distribution and internal linking patterns..."):
        
        # Calculate section analysis
        section_pr = defaultdict(float)
        section_counts = defaultdict(int)
        for url, score in pagerank_scores.items():
            section = section_mapping.get(url, 'Other')
            section_pr[section] += score
            section_counts[section] += 1
        
        total_pr = sum(section_pr.values())
        section_analysis = [
            {
                'section': section,
                'pagerank': score,
                'percentage': score/total_pr*100,
                'page_count': section_counts[section],
                'avg_pagerank': score/section_counts[section]
            }
            for section, score in sorted(section_pr.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Get top pages
        top_pages = [
            {
                'url': url,
                'pagerank': score,
                'section': section_mapping.get(url, 'Other')
            }
            for url, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        ]
        
        # Priority analysis
        priority_analysis = []
        if prio_df is not None and not prio_df.empty:
            for _, row in prio_df.iterrows():
                url = str(row['url'])
                priority_analysis.append({
                    'url': url,
                    'pagerank': pagerank_scores.get(url, 0.0),
                    'category': str(row['category']),
                    'keyword': str(row['keyword']),
                    'rank_position': next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == url), len(pagerank_scores))
                })
        
        # Analyze linking matrix and anchor texts
        linking_matrix = analyze_section_linking_matrix(graph, section_mapping)
        anchor_analysis = analyze_anchor_texts_by_section(anchor_texts, section_mapping)
        
        # Get AI recommendations
        ai_recommendations = get_ai_recommendations(
            section_analysis, top_pages, priority_analysis, 
            linking_matrix, anchor_analysis, openai_key
        )
    
    # Display AI recommendations
    st.markdown(f"""
    <div class="ai-insight">
    <h3>🎯 AI-Generated SEO Strategy & Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(ai_recommendations)
    
    # Additional AI-powered insights
    st.subheader("📈 Advanced Performance Metrics")
    
    # Calculate advanced metrics
    total_pages = len(pagerank_scores)
    top_10_pr = sum([score for _, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]])
    concentration_ratio = (top_10_pr / sum(pagerank_scores.values()) * 100) if sum(pagerank_scores.values()) > 0 else 0
    
    # Section diversity
    section_count = len(set(section_mapping.values()))
    avg_section_size = total_pages / section_count if section_count > 0 else 0
    
    # Linking efficiency
    total_internal_links = graph.number_of_edges()
    link_density = total_internal_links / (total_pages * (total_pages - 1)) if total_pages > 1 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
        <h4>🎯 Distribution Analysis</h4>
        <p><strong>Total Pages:</strong> {total_pages:,}</p>
        <p><strong>Top 10 Concentration:</strong> {concentration_ratio:.1f}%</p>
        <p><strong>Assessment:</strong> {'Highly concentrated' if concentration_ratio > 50 else 'Well distributed' if concentration_ratio < 30 else 'Moderately concentrated'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
        <h4>🏗️ Site Structure</h4>
        <p><strong>Sections:</strong> {section_count}</p>
        <p><strong>Avg Pages/Section:</strong> {avg_section_size:.1f}</p>
        <p><strong>Structure:</strong> {'Well organized' if 5 <= section_count <= 15 else 'Consider reorganizing'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
        <h4>🔗 Linking Efficiency</h4>
        <p><strong>Total Internal Links:</strong> {total_internal_links:,}</p>
        <p><strong>Link Density:</strong> {link_density*100:.3f}%</p>
        <p><strong>Efficiency:</strong> {'Optimal' if 0.1 <= link_density*100 <= 2 else 'Needs optimization'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Priority page performance summary
    if priority_analysis:
        st.subheader("🎯 Priority Page Performance Summary")
        
        top_20_urls = set([url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:20]])
        priority_in_top_20 = sum(1 for p in priority_analysis if p['url'] in top_20_urls)
        avg_priority_rank = np.mean([p['rank_position'] for p in priority_analysis])
        
        performance_score = (priority_in_top_20 / len(priority_analysis) * 50) + max(0, (100 - avg_priority_rank) / 100 * 50)
        
        st.markdown(f"""
        <div class="{'success-box' if performance_score > 70 else 'warning-box' if performance_score > 40 else 'low-performance'}">
        <h4>📊 Priority Page Performance Score: {performance_score:.1f}/100</h4>
        <p><strong>Pages in Top 20:</strong> {priority_in_top_20}/{len(priority_analysis)} ({priority_in_top_20/len(priority_analysis)*100:.1f}%)</p>
        <p><strong>Average Rank:</strong> #{avg_priority_rank:.0f}</p>
        <p><strong>Status:</strong> {'Excellent' if performance_score > 80 else 'Good' if performance_score > 60 else 'Needs Improvement' if performance_score > 40 else 'Requires Urgent Attention'}</p>
        </div>
        """, unsafe_allow_html=True)

###############################################################################
#                              Main Application                               #
###############################################################################

def process_and_display_analysis(crawled_urls, page_data, graph, anchor_texts, priority_pages_df, alpha, openai_key):
    """Process crawled data and display comprehensive analysis"""
    
    # Calculate PageRank safely
    with st.spinner("Calculating PageRank scores..."):
        pagerank_scores = calculate_pagerank_safe(graph, alpha)
    
    if not pagerank_scores:
        st.error("Failed to calculate PageRank scores")
        return
    
    st.success(f"✅ PageRank calculated for {len(pagerank_scores):,} pages")
    
    # Create section mapping using URL structure
    section_mapping = {}
    for url in crawled_urls:
        section_mapping[url] = categorize_url_by_structure(url)
    
    # Advanced analytics
    with st.spinner("Performing advanced analytics..."):
        section_matrix = analyze_section_linking_matrix(graph, section_mapping)
        anchor_analysis = analyze_anchor_texts_by_section(anchor_texts, section_mapping)
    
    # Display results in comprehensive tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Section Analysis",
        "🏆 Top Pages", 
        "🎯 Priority Pages",
        "🔍 Five Critical Questions",
        "🔗 Anchor Text Analysis",
        "🤖 AI Insights"
    ])
    
    with tab1:
        tab_section_analysis(pagerank_scores, section_mapping, anchor_analysis, section_matrix, openai_key)
    
    with tab2:
        tab_top_pages_analysis(pagerank_scores, section_mapping, page_data)
    
    with tab3:
        tab_priority_pages_analysis(pagerank_scores, section_mapping, priority_pages_df, graph, anchor_texts)
    
    with tab4:
        tab_five_critical_questions(pagerank_scores, section_mapping, priority_pages_df, anchor_analysis, graph)
    
    with tab5:
        tab_anchor_text_analysis(anchor_texts, section_mapping, priority_pages_df)
    
    with tab6:
        tab_ai_insights(priority_pages_df, pagerank_scores, section_mapping, anchor_texts, graph, openai_key)
    
    # Comprehensive export functionality
    st.header("📥 Export Comprehensive Analysis")
    
    # Prepare comprehensive export data
    export_data = []
    for url, score in pagerank_scores.items():
        page_info = page_data.get(url, {})
        section = section_mapping.get(url, 'Other')
        
        # Get ranking position
        rank_position = next((i+1 for i, (u, _) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)) if u == url), len(pagerank_scores))
        
        # Get anchor text data
        anchors = anchor_texts.get(url, Counter())
        top_anchor = anchors.most_common(1) if anchors else ''
        
        export_data.append({
            'URL': url,
            'PageRank': score,
            'Rank_Position': rank_position,
            'Section': section,
            'Title': page_info.get('title', ''),
            'Meta_Description': page_info.get('meta_description', ''),
            'H1': page_info.get('h1', ''),
            'Word_Count': page_info.get('word_count', 0),
            'Internal_Links_Out': page_info.get('internal_links', 0),
            'Internal_Links_In': len(list(graph.predecessors(url))) if graph.has_node(url) else 0,
            'Route_Depth': page_info.get('route_depth', 0),
            'Status_Code': page_info.get('status_code', ''),
            'Content_Length': page_info.get('content_length', 0),
            'Images': page_info.get('images', 0),
            'Headers': page_info.get('headers', 0),
            'Top_Anchor_Text': top_anchor,
            'Total_Anchor_Instances': sum(anchors.values()),
            'Unique_Anchor_Texts': len(anchors),
        })

    export_df = pd.DataFrame(export_data)
    export_df = fix_dataframe_types(export_df)
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full PageRank Analysis (CSV)",
        data=csv_data,
        file_name=f"pagerank_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

###############################################################################
#                                App Main                                     #
###############################################################################

def main():
    st.title("🚀 Advanced PageRank SEO Analyzer")
    st.markdown("**Ultimate Edition with Enhanced AI Recommendations and Visual Analytics**")

    st.sidebar.header("🔧 Analysis Configuration")
    analysis_mode = st.sidebar.radio(
        "Analysis Mode", ["Website Crawler", "Screaming Frog CSV"]
    )
    alpha = st.sidebar.slider("PageRank Alpha (damping factor)", 0.1, 0.99, 0.85, 0.01)
    openai_api_key = st.sidebar.text_input("OpenAI API Key (for AI Insights)", type="password", help="Get an API key at https://platform.openai.com/")

    priority_file = st.sidebar.file_uploader("Upload Priority Pages CSV", type=['csv'])
    priority_pages_df = None
    if priority_file:
        priority_pages_df = load_and_process_priority_csv(priority_file)

    if analysis_mode == "Website Crawler":
        st.sidebar.header("Crawl Settings")
        seed_url = st.sidebar.text_input("Website URL", "https://example.com")
        max_pages = st.sidebar.slider("Max Pages", 100, 10000, 5000, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 5)
        delay = st.sidebar.slider("Delay Between Requests (seconds)", 0.1, 3.0, 0.2, 0.1)

        go_crawl = st.sidebar.button("Start Crawl and Analyze", type='primary')
        if go_crawl:
            st.info(f"Crawling {seed_url} (max {max_pages}, depth {max_depth})")
            crawler = OptimizedCrawler(seed_url, max_pages, max_depth, delay)
            crawled_urls = crawler.crawl()
            page_data = crawler.page_data
            anchor_texts = crawler.anchor_texts
            graph = crawler.graph
            process_and_display_analysis(
                crawled_urls, page_data, graph, anchor_texts,
                priority_pages_df, alpha, openai_api_key
            )
    else:
        crawl_file = st.sidebar.file_uploader("Upload Screaming Frog CSV", type=["csv"])
        if crawl_file:
            page_data, graph, anchor_texts = load_screaming_frog_csv(crawl_file)
            crawled_urls = list(page_data.keys())
            if len(crawled_urls) == 0:
                st.error("No URLs found in your Screaming Frog CSV. Please check your mapping and file.")
                return
            process_and_display_analysis(
                crawled_urls, page_data, graph, anchor_texts,
                priority_pages_df, alpha, openai_api_key
            )
        else:
            st.info("Please upload a Screaming Frog CSV to begin.")

    st.markdown("<br><hr><center><small>🚀 Advanced PageRank SEO Analyzer &copy; 2025 | Powered by Streamlit</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

