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
import tempfile
import xlsxwriter
import openai
from openai import OpenAI
import os
import gc
from functools import lru_cache
import logging
import io

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Configure Streamlit
st.set_page_config(
    page_title="🚀 PageRank SEO Analyzer - Complete Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #e8f4fd 0%, #dbeafe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.15);
    }
    
    .success-card {
        background: linear-gradient(135deg, #dcfce7 0%, #86efac 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #22c55e;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.15);
    }
    
    .question-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

class OptimizedCrawler:
    """Memory-efficient web crawler"""
    
    def __init__(self, seed_url, max_pages=1000, max_depth=3, delay=0.1):
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
            'end_time': None
        }
    
    def crawl(self):
        """Execute optimized crawling"""
        visited = set()
        to_visit = [(self.seed_url, 0)]
        domain = urlparse(self.seed_url).netloc
        
        self.crawl_stats['start_time'] = datetime.now()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while to_visit and len(visited) < self.max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited or depth > self.max_depth:
                continue
            
            try:
                # Update progress
                progress = len(visited) / self.max_pages
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Crawling: {len(visited)}/{self.max_pages} pages | Current: {current_url[:60]}...")
                
                # Make request
                response = self.session.get(current_url, timeout=6)
                
                if 'text/html' not in response.headers.get('content-type', '').lower():
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract page data
                title = soup.title.string.strip() if soup.title else ''
                h1 = soup.h1.get_text().strip() if soup.h1 else ''
                
                meta_desc = ''
                meta_tag = soup.find('meta', attrs={'name': 'description'})
                if meta_tag:
                    meta_desc = meta_tag.get('content', '')
                
                self.page_data[current_url] = {
                    'title': title[:150],
                    'h1': h1[:100],
                    'meta_description': meta_desc[:200],
                    'word_count': len(soup.get_text().split()),
                    'route_depth': len([seg for seg in urlparse(current_url).path.split('/') if seg]),
                    'internal_links': 0,
                    'external_links': 0
                }
                
                # Extract links
                internal_links = 0
                external_links = 0
                new_urls = []
                
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '').strip()
                    
                    if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                        continue
                    
                    try:
                        full_url = urljoin(current_url, href).split('#')[0]
                        
                        if urlparse(full_url).netloc == domain:
                            internal_links += 1
                            self.graph.add_edge(current_url, full_url)
                            self.crawl_stats['links_found'] += 1
                            
                            # Extract anchor text
                            anchor_text = link.get_text().strip()[:100]
                            if anchor_text:
                                self.anchor_texts[full_url][anchor_text] += 1
                            
                            # Add to crawl queue
                            if (full_url not in visited and
                                (full_url, depth + 1) not in to_visit and
                                full_url not in new_urls and
                                len(visited) + len(to_visit) < self.max_pages):
                                new_urls.append(full_url)
                        else:
                            external_links += 1
                    except:
                        continue
                
                # Add new URLs to queue (limit to prevent memory issues)
                for new_url in new_urls[:15]:
                    to_visit.append((new_url, depth + 1))
                
                self.page_data[current_url]['internal_links'] = internal_links
                self.page_data[current_url]['external_links'] = external_links
                
                visited.add(current_url)
                self.crawl_stats['pages_crawled'] = len(visited)
                
                time.sleep(self.delay)
                
            except Exception as e:
                self.crawl_stats['errors'] += 1
                continue
        
        self.crawl_stats['end_time'] = datetime.now()
        progress_bar.progress(1.0)
        status_text.text(f"✅ Crawling complete! {len(visited)} pages crawled.")
        
        return visited

class SectionAnalyzer:
    """Optimized section categorization"""
    
    def __init__(self):
        self.categories = {
            'homepage': ['home', 'index', 'main'],
            'content': ['blog', 'news', 'article', 'post'],
            'product': ['product', 'shop', 'store', 'buy'],
            'service': ['service', 'services', 'solutions'],
            'company': ['about', 'company', 'team'],
            'contact': ['contact', 'location'],
            'category': ['category', 'tag', 'tags'],
            'author': ['author', 'writer'],
            'finance': ['loan', 'loans', 'finance', 'bank'],
            'institution': ['institution', 'university', 'school'],
            'legal': ['privacy', 'terms', 'legal']
        }
        
        self.business_value = {
            'high': ['homepage', 'product', 'service', 'finance', 'institution'],
            'medium': ['content', 'company', 'author'],
            'low': ['category', 'legal', 'contact']
        }
    
    @lru_cache(maxsize=500)
    def categorize_url(self, url, title="", h1=""):
        """Categorize URL into sections"""
        path = urlparse(url).path.lower()
        text = f"{path} {title[:50]} {h1[:50]}".lower()
        
        if not path or path in ['/', '']:
            return 'homepage'
        
        # Pattern matching
        if 'tag' in path or 'category' in path:
            return 'category'
        if 'author' in path:
            return 'author'
        if 'news' in path or 'blog' in path:
            return 'content'
        if 'loan' in path or 'finance' in path:
            return 'finance'
        
        # Keyword scoring
        scores = {}
        for category, keywords in self.categories.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to first path segment
        segments = [seg for seg in path.split('/') if seg]
        return segments[0] if segments else 'other'
    
    def get_business_value(self, category):
        """Get business value for category"""
        for value, categories in self.business_value.items():
            if category in categories:
                return value
        return 'medium'

class StandardVisualizer:
    """Create standard, optimized charts including network graph - COMPLETE VERSION"""
    
    def __init__(self, pagerank_scores, section_mapping, section_analyzer, graph, page_data):
        self.pagerank_scores = pagerank_scores
        self.section_mapping = section_mapping
        self.section_analyzer = section_analyzer
        self.graph = graph
        self.page_data = page_data
    
    def create_section_bar_chart(self):
        """Standard section PageRank bar chart"""
        section_pr = defaultdict(float)
        for url, score in self.pagerank_scores.items():
            section = self.section_mapping.get(url, 'other')
            section_pr[section] += score
        
        total_pr = sum(section_pr.values())
        
        data = []
        for section, pr_score in sorted(section_pr.items(), key=lambda x: x[1], reverse=True):
            business_value = self.section_analyzer.get_business_value(section)
            percentage = (pr_score / total_pr * 100) if total_pr > 0 else 0
            
            data.append({
                'section': section,
                'pagerank': pr_score,
                'percentage': percentage,
                'business_value': business_value
            })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='section',
            y='pagerank',
            color='business_value',
            color_discrete_map={
                'high': '#22c55e',
                'medium': '#f59e0b',
                'low': '#ef4444'
            },
            title='📊 PageRank Distribution by Section',
            labels={'pagerank': 'PageRank Score', 'section': 'Section'},
            text='percentage'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=500, showlegend=True)
        
        return fig
    
    def create_top_pages_chart(self):
        """Standard top pages bar chart"""
        top_pages = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        data = []
        for url, score in top_pages:
            section = self.section_mapping.get(url, 'other')
            page_name = urlparse(url).path.split('/')[-1] or 'Homepage'
            
            data.append({
                'page': page_name[:20] + '...' if len(page_name) > 20 else page_name,
                'pagerank': score,
                'section': section,
                'url': url
            })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='page',
            y='pagerank',
            color='section',
            title='🏆 Top 15 Pages by PageRank',
            labels={'pagerank': 'PageRank Score', 'page': 'Page'},
            hover_data=['url']
        )
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        
        return fig
    
    def create_business_value_pie(self):
        """Standard business value pie chart - FIXED METHOD"""
        business_values = {'high': 0, 'medium': 0, 'low': 0}
        
        for url, score in self.pagerank_scores.items():
            section = self.section_mapping.get(url, 'other')
            business_value = self.section_analyzer.get_business_value(section)
            business_values[business_value] += score
        
        fig = px.pie(
            values=list(business_values.values()),
            names=['High Value', 'Medium Value', 'Low Value'],
            title='💼 PageRank Distribution by Business Value',
            color_discrete_map={
                'High Value': '#22c55e',
                'Medium Value': '#f59e0b',
                'Low Value': '#ef4444'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def create_network_graph(self):
        """Create interactive network graph visualization - FIXED"""
        try:
            # Get top 25 pages for network visualization
            top_pages = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:25]
            top_urls = [url for url, _ in top_pages]
            
            # Create subgraph
            subgraph = self.graph.subgraph(top_urls)
            
            if len(subgraph.nodes()) == 0:
                return go.Figure()
            
            # Create layout
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50)
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in subgraph.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='rgba(125, 125, 125, 0.6)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            node_hover = []
            
            for node in subgraph.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Get node properties
                    pr_score = self.pagerank_scores.get(node, 0)
                    section = self.section_mapping.get(node, 'other')
                    business_value = self.section_analyzer.get_business_value(section)
                    
                    # Size based on PageRank
                    node_size.append(max(10, pr_score * 1000))
                    
                    # Color based on business value
                    if business_value == 'high':
                        node_color.append('#22c55e')
                    elif business_value == 'medium':
                        node_color.append('#f59e0b')
                    else:
                        node_color.append('#ef4444')
                    
                    # Page name for display
                    page_name = urlparse(node).path.split('/')[-1] or 'Home'
                    node_text.append(page_name[:15] + '...' if len(page_name) > 15 else page_name)
                    
                    # Hover info
                    title = ''
                    if node in self.page_data:
                        title = self.page_data[node].get('title', '')
                    
                    node_hover.append(
                        f"<b>{page_name}</b><br>" +
                        f"PageRank: {pr_score:.4f}<br>" +
                        f"Section: {section}<br>" +
                        f"Business Value: {business_value}<br>" +
                        f"Title: {title[:50]}..." if len(title) > 50 else f"Title: {title}"
                    )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_hover,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                marker=dict(
                    size=node_size,
                    color=node_color,
                    opacity=0.8,
                    line=dict(width=2, color="white")
                ),
                showlegend=False
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace])
            
            # Update layout with modern syntax
            fig.update_layout(
                title={
                    'text': '🕸️ PageRank Flow Network (Top 25 Pages)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text="Node size = PageRank score | Color = Business value (Green=High, Yellow=Medium, Red=Low)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=10)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating network graph: {str(e)}")
            return go.Figure()
    
    def create_section_matrix_heatmap(self):
        """Standard section linking heatmap"""
        section_links = defaultdict(lambda: defaultdict(int))
        
        for source, target in self.graph.edges():
            source_section = self.section_mapping.get(source, 'other')
            target_section = self.section_mapping.get(target, 'other')
            section_links[source_section][target_section] += 1
        
        sections = sorted(set(self.section_mapping.values()))
        
        # Create matrix
        matrix = []
        for source in sections:
            row = []
            for target in sections:
                row.append(section_links[source][target])
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=sections,
            y=sections,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='🔥 Section-to-Section Linking Heatmap',
            xaxis_title='Target Section',
            yaxis_title='Source Section',
            height=500
        )
        
        return fig
    
    def create_route_depth_chart(self):
        """Standard route depth analysis"""
        depth_data = defaultdict(list)
        
        for url, data in self.page_data.items():
            depth = data.get('route_depth', 0)
            pr_score = self.pagerank_scores.get(url, 0)
            depth_data[depth].append(pr_score)
        
        chart_data = []
        for depth, scores in depth_data.items():
            if scores:
                chart_data.append({
                    'depth': depth,
                    'avg_pagerank': np.mean(scores),
                    'page_count': len(scores),
                    'total_pagerank': sum(scores)
                })
        
        df = pd.DataFrame(chart_data)
        
        if df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average PageRank by Depth', 'Page Count by Depth')
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['depth'],
                y=df['avg_pagerank'],
                mode='lines+markers',
                name='Avg PageRank',
                line=dict(color='#3b82f6', width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df['depth'],
                y=df['page_count'],
                name='Page Count',
                marker_color='#10b981'
            ),
            row=1, col=2
        )
        
        fig.update_layout(title='📊 Route Depth Analysis', height=400)
        
        return fig

def handle_csv_upload_with_mapping():
    """Enhanced CSV upload with column mapping - FIXED for PyArrow compatibility"""
    st.subheader("📁 Priority Pages & CSV Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload your priority pages, Ahrefs data, or any relevant CSV"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! {len(df)} rows, {len(df.columns)} columns")
            
            # Show preview
            st.markdown("**📊 File Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping interface
            st.markdown("**🔧 Column Mapping:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                url_column = st.selectbox(
                    "URL Column",
                    options=['None'] + df.columns.tolist(),
                    help="Select the column containing URLs"
                )
            
            with col2:
                keyword_column = st.selectbox(
                    "Keywords Column (Optional)",
                    options=['None'] + df.columns.tolist(),
                    help="Select the column with target keywords"
                )
            
            with col3:
                ranking_column = st.selectbox(
                    "Rankings/Metrics Column (Optional)",
                    options=['None'] + df.columns.tolist(),
                    help="Select column with rankings, traffic, or other metrics"
                )
            
            # Additional options
            st.markdown("**⚙️ Processing Options:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                remove_duplicates = st.checkbox("Remove duplicate URLs", value=True)
                clean_urls = st.checkbox("Clean and normalize URLs", value=True)
            
            with col2:
                limit_rows = st.checkbox("Limit rows (for performance)")
                if limit_rows:
                    max_rows = st.slider("Max rows to process", 100, 5000, 1000)
                else:
                    max_rows = len(df)
            
            # Process the data
            if url_column != 'None':
                processed_df = df.copy()
                
                # Limit rows if specified
                if limit_rows:
                    processed_df = processed_df.head(max_rows)
                
                # Create standardized dataframe with proper data types - FIXED for PyArrow
                result_df = pd.DataFrame()
                result_df['URL'] = processed_df[url_column].astype(str)  # Ensure string type
                
                if keyword_column != 'None':
                    result_df['Target Keywords'] = processed_df[keyword_column].astype(str)
                else:
                    result_df['Target Keywords'] = ''
                
                if ranking_column != 'None':
                    # Convert to numeric, handle non-numeric values
                    try:
                        result_df['Metrics'] = pd.to_numeric(processed_df[ranking_column], errors='coerce').fillna(0).astype(int)
                    except:
                        result_df['Metrics'] = processed_df[ranking_column].astype(str)
                else:
                    result_df['Metrics'] = 0
                
                # Clean URLs if requested
                if clean_urls:
                    result_df['URL'] = result_df['URL'].str.strip()
                    result_df['URL'] = result_df['URL'].str.replace(r'^https?://', '', regex=True)
                    result_df['URL'] = 'https://' + result_df['URL']
                
                # Remove duplicates if requested
                if remove_duplicates:
                    initial_count = len(result_df)
                    result_df = result_df.drop_duplicates(subset=['URL'])
                    final_count = len(result_df)
                    if initial_count != final_count:
                        st.info(f"Removed {initial_count - final_count} duplicate URLs")
                
                # Ensure all columns have consistent data types for PyArrow compatibility
                result_df['URL'] = result_df['URL'].astype(str)
                result_df['Target Keywords'] = result_df['Target Keywords'].astype(str)
                
                st.markdown("**✅ Processed Data Preview:**")
                st.dataframe(result_df.head(), use_container_width=True)
                
                st.success(f"✅ Successfully processed {len(result_df)} records")
                
                return result_df
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None
    
    return None

def generate_ai_recommendations(pagerank_scores, section_stats, priority_analysis, openai_key):
    """Generate OpenAI recommendations"""
    if not openai_key:
        return "OpenAI API key not provided."
    
    try:
        client = OpenAI(api_key=openai_key)
        
        # Prepare analysis summary
        total_pages = len(pagerank_scores)
        top_sections = section_stats[:5] if section_stats else []
        
        # Calculate key metrics
        total_pr = sum(s['total_pr'] for s in section_stats) if section_stats else 1
        high_value_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'high') if section_stats else 0
        low_value_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'low') if section_stats else 0
        
        efficiency_score = (high_value_pr / total_pr * 100) if total_pr > 0 else 0
        waste_percentage = (low_value_pr / total_pr * 100) if total_pr > 0 else 0
        
        prompt = f"""
        As an expert SEO consultant, analyze this PageRank data and provide actionable recommendations:
        
        **Website Analysis Summary:**
        - Total pages analyzed: {total_pages}
        - PageRank efficiency score: {efficiency_score:.1f}%
        - PageRank waste in low-value sections: {waste_percentage:.1f}%
        
        **Top Performing Sections:**
        {chr(10).join([f"- {s['section']}: {s['total_pr']:.4f} PageRank ({s['percentage']:.1f}%) - {s['business_value']} value" for s in top_sections])}
        
        **Priority Pages Analysis:**
        {f"Found {len([p for p in priority_analysis if p['found']])} of {len(priority_analysis)} priority pages in crawl" if priority_analysis else "No priority pages provided"}
        
        Provide specific, actionable recommendations in these areas:
        
        1. **Immediate Quick Wins (0-2 weeks)**
        2. **Strategic Optimizations (1-3 months)**  
        3. **Technical Implementation Steps**
        4. **Expected Impact & ROI**
        
        Keep recommendations practical and implementation-focused.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior SEO strategist specializing in technical SEO and PageRank optimization. Provide specific, actionable recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating AI recommendations: {str(e)}"

def generate_excel_report(crawler, pagerank_scores, section_mapping, section_analyzer, priority_df=None):
    """Generate optimized Excel report with proper error handling"""
    try:
        # Create a BytesIO object to store the Excel file in memory
        output = io.BytesIO()
        
        # Create workbook and worksheet
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#667eea',
            'font_color': 'white',
            'align': 'center',
            'border': 1
        })
        
        data_format = workbook.add_format({
            'border': 1,
            'align': 'left'
        })
        
        number_format = workbook.add_format({
            'num_format': '0.000000',
            'border': 1,
            'align': 'right'
        })
        
        percentage_format = workbook.add_format({
            'num_format': '0.00%',
            'border': 1,
            'align': 'right'
        })
        
        # Executive Summary Sheet
        summary_sheet = workbook.add_worksheet('Executive Summary')
        summary_sheet.set_column('A:A', 30)
        summary_sheet.set_column('B:B', 20)
        
        # Add title
        summary_sheet.merge_range('A1:B1', 'PageRank SEO Analysis Report', header_format)
        
        # Summary data
        summary_data = [
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Website URL', crawler.seed_url],
            ['Total Pages Analyzed', len(pagerank_scores)],
            ['Total Internal Links', len(crawler.graph.edges())],
            ['Unique Sections Found', len(set(section_mapping.values()))],
            ['Crawl Errors', crawler.crawl_stats.get('errors', 0)],
            ['Analysis Duration (seconds)', 
             (crawler.crawl_stats['end_time'] - crawler.crawl_stats['start_time']).total_seconds() if crawler.crawl_stats.get('end_time') and crawler.crawl_stats.get('start_time') else 0]
        ]
        
        for row, (metric, value) in enumerate(summary_data, 2):
            summary_sheet.write(row, 0, metric, data_format)
            summary_sheet.write(row, 1, value, data_format)
        
        # Top Pages Sheet
        pages_sheet = workbook.add_worksheet('Top Pages Analysis')
        pages_sheet.set_column('A:A', 50)  # URL column
        pages_sheet.set_column('B:B', 15)  # PageRank column
        pages_sheet.set_column('C:C', 15)  # Section column
        pages_sheet.set_column('D:D', 15)  # Business Value column
        pages_sheet.set_column('E:E', 40)  # Title column
        pages_sheet.set_column('F:F', 10)  # Depth column
        pages_sheet.set_column('G:G', 15)  # Internal Links column
        
        # Headers
        headers = ['URL', 'PageRank Score', 'Section', 'Business Value', 'Page Title', 'Route Depth', 'Internal Links']
        for col, header in enumerate(headers):
            pages_sheet.write(0, col, header, header_format)
        
        # Top 100 pages data
        top_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        
        for row, (url, score) in enumerate(top_pages, 1):
            section = section_mapping.get(url, 'other')
            business_value = section_analyzer.get_business_value(section)
            page_data = crawler.page_data.get(url, {})
            title = page_data.get('title', 'No Title')
            route_depth = page_data.get('route_depth', 0)
            internal_links = page_data.get('internal_links', 0)
            
            pages_sheet.write(row, 0, url, data_format)
            pages_sheet.write(row, 1, score, number_format)
            pages_sheet.write(row, 2, section, data_format)
            pages_sheet.write(row, 3, business_value, data_format)
            pages_sheet.write(row, 4, title, data_format)
            pages_sheet.write(row, 5, route_depth, data_format)
            pages_sheet.write(row, 6, internal_links, data_format)
        
        # Section Analysis Sheet
        sections_sheet = workbook.add_worksheet('Section Analysis')
        sections_sheet.set_column('A:F', 20)
        
        section_headers = ['Section', 'Total PageRank', 'Page Count', 'Average PageRank', 'Business Value', 'Percentage of Total PR']
        for col, header in enumerate(section_headers):
            sections_sheet.write(0, col, header, header_format)
        
        # Calculate section statistics
        section_stats = defaultdict(lambda: {'pr': 0, 'count': 0})
        total_pr = sum(pagerank_scores.values())
        
        for url, section in section_mapping.items():
            pr_score = pagerank_scores.get(url, 0)
            section_stats[section]['pr'] += pr_score
            section_stats[section]['count'] += 1
        
        # Write section data
        for row, (section, stats) in enumerate(sorted(section_stats.items(), key=lambda x: x[1]['pr'], reverse=True), 1):
            business_value = section_analyzer.get_business_value(section)
            avg_pr = stats['pr'] / stats['count'] if stats['count'] > 0 else 0
            percentage = stats['pr'] / total_pr if total_pr > 0 else 0
            
            sections_sheet.write(row, 0, section, data_format)
            sections_sheet.write(row, 1, stats['pr'], number_format)
            sections_sheet.write(row, 2, stats['count'], data_format)
            sections_sheet.write(row, 3, avg_pr, number_format)
            sections_sheet.write(row, 4, business_value, data_format)
            sections_sheet.write(row, 5, percentage, percentage_format)
        
        # Priority Pages Sheet (if provided)
        if priority_df is not None and not priority_df.empty:
            priority_sheet = workbook.add_worksheet('Priority Pages Analysis')
            priority_sheet.set_column('A:A', 50)
            priority_sheet.set_column('B:F', 20)
            
            priority_headers = ['Priority URL', 'PageRank Score', 'Found in Crawl', 'Section', 'Business Value', 'Target Keywords']
            for col, header in enumerate(priority_headers):
                priority_sheet.write(0, col, header, header_format)
            
            # Process priority pages
            for row, (_, page_row) in enumerate(priority_df.iterrows(), 1):
                url = page_row.get('URL', '')
                keywords = str(page_row.get('Target Keywords', ''))
                
                if url in pagerank_scores:
                    score = pagerank_scores[url]
                    found = 'Yes'
                    section = section_mapping.get(url, 'other')
                    business_value = section_analyzer.get_business_value(section)
                else:
                    score = 0
                    found = 'No'
                    section = 'Not Found'
                    business_value = 'Unknown'
                
                priority_sheet.write(row, 0, url, data_format)
                priority_sheet.write(row, 1, score, number_format if found == 'Yes' else data_format)
                priority_sheet.write(row, 2, found, data_format)
                priority_sheet.write(row, 3, section, data_format)
                priority_sheet.write(row, 4, business_value, data_format)
                priority_sheet.write(row, 5, keywords, data_format)
        
        # Close workbook and get the data
        workbook.close()
        
        # Get the data from BytesIO
        output.seek(0)
        excel_data = output.read()
        output.close()
        
        return excel_data
        
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")
        return None

def main():
    """Main application - COMPLETE AND FIXED"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 PageRank SEO Analyzer - Complete Edition</h1>
        <h2>Network Graph • Fixed Excel Export • AI Insights • CSV Mapping</h2>
        <p>Comprehensive analysis of the 5 critical PageRank distribution questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("### 🔧 Configuration")
    
    # OpenAI API Key
    openai_key = st.sidebar.text_input(
        "🤖 OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for AI recommendations"
    )
    
    if openai_key:
        st.sidebar.success("✅ AI recommendations enabled")
    
    # Website URL
    website_url = st.sidebar.text_input(
        "🌐 Website URL",
        placeholder="https://example.com"
    )
    
    # Crawling settings
    st.sidebar.markdown("### ⚙️ Crawling Settings")
    max_pages = st.sidebar.slider("📄 Max Pages", 100, 1000, 300, step=50)
    max_depth = st.sidebar.slider("🕳️ Max Depth", 1, 4, 3)
    crawl_delay = st.sidebar.slider("⏱️ Delay (sec)", 0.1, 1.0, 0.2, step=0.1)
    
    # CSV Upload with mapping
    priority_df = handle_csv_upload_with_mapping()
    
    # Main analysis
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if not website_url:
            st.error("❌ Please enter a website URL")
            return
        
        # Initialize
        crawler = OptimizedCrawler(website_url, max_pages, max_depth, crawl_delay)
        section_analyzer = SectionAnalyzer()
        
        # Step 1: Crawl
        st.markdown("## 🕷️ Website Crawling")
        crawled_urls = crawler.crawl()
        
        if not crawled_urls:
            st.error("❌ No pages crawled")
            return
        
        st.success(f"✅ Crawled {len(crawled_urls)} pages with {crawler.crawl_stats['links_found']} links")
        
        # Step 2: Categorize
        st.markdown("## 🔍 Section Detection")
        section_mapping = {}
        
        with st.spinner("Categorizing pages..."):
            for url in crawled_urls:
                page_data = crawler.page_data.get(url, {})
                title = page_data.get('title', '')
                h1 = page_data.get('h1', '')
                section = section_analyzer.categorize_url(url, title, h1)
                section_mapping[url] = section
        
        st.success(f"✅ Identified {len(set(section_mapping.values()))} sections")
        
        # Step 3: Calculate PageRank
        st.markdown("## 📊 PageRank Calculation")
        
        with st.spinner("Computing PageRank..."):
            if len(crawler.graph.edges()) == 0:
                # No links - equal distribution
                num_pages = len(crawled_urls)
                pagerank_scores = {url: 1.0/num_pages for url in crawled_urls}
            else:
                pagerank_scores = nx.pagerank(crawler.graph, alpha=0.85, max_iter=100)
                # Normalize
                total_score = sum(pagerank_scores.values())
                pagerank_scores = {url: score/total_score for url, score in pagerank_scores.items()}
        
        st.success(f"✅ PageRank calculated for {len(pagerank_scores)} pages")
        
        # Step 4: Analysis
        st.markdown("## 📈 Analysis Results")
        
        # Calculate section statistics
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
            percentage = (pr_score / total_pr * 100) if total_pr > 0 else 0
            
            section_stats.append({
                'section': section,
                'total_pr': pr_score,
                'page_count': section_count[section],
                'percentage': percentage,
                'business_value': business_value
            })
        
        section_stats.sort(key=lambda x: x['total_pr'], reverse=True)
        
        # Priority analysis
        priority_analysis = None
        if priority_df is not None and not priority_df.empty:
            priority_analysis = []
            for _, row in priority_df.iterrows():
                url = row.get('URL', '')
                keywords = row.get('Target Keywords', '')
                
                if url in pagerank_scores:
                    score = pagerank_scores[url]
                    section = section_mapping.get(url, 'other')
                    priority_analysis.append({
                        'url': url,
                        'pagerank': score,
                        'section': section,
                        'keywords': keywords,
                        'found': True
                    })
                else:
                    priority_analysis.append({
                        'url': url,
                        'pagerank': 0,
                        'section': 'Unknown',
                        'keywords': keywords,
                        'found': False
                    })
        
        # Create visualizations - FIXED: Pass all required parameters
        visualizer = StandardVisualizer(pagerank_scores, section_mapping, section_analyzer, crawler.graph, crawler.page_data)
        
        # Display in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 5 Key Questions", "📊 Standard Charts", "🕸️ Network Graph", "🤖 AI Recommendations", "📋 Export"
        ])
        
        with tab1:
            st.markdown("## 🎯 The 5 Critical Questions")
            
            # Question 1
            st.markdown('<div class="question-header">1. Which Sections receive the most PageRank?</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_sections = visualizer.create_section_bar_chart()
                st.plotly_chart(fig_sections, use_container_width=True)
            
            with col2:
                st.markdown("### 🏆 Top Sections")
                for i, stats in enumerate(section_stats[:5], 1):
                    emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[stats['business_value']]
                    st.markdown(f"""
                    **{i}. {stats['section'].title()}** {emoji}  
                    📊 {stats['total_pr']:.4f} ({stats['percentage']:.1f}%)  
                    📄 {stats['page_count']} pages
                    """)
            
            # Question 2
            st.markdown('<div class="question-header">2. Which specific pages receive the most PageRank?</div>', unsafe_allow_html=True)
            
            fig_top_pages = visualizer.create_top_pages_chart()
            st.plotly_chart(fig_top_pages, use_container_width=True)
            
            # Top pages table - FIXED: Ensure proper data types
            top_pages_data = []
            for i, (url, score) in enumerate(sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:15], 1):
                page_data = crawler.page_data.get(url, {})
                section = section_mapping.get(url, 'other')
                
                top_pages_data.append({
                    'Rank': int(i),
                    'URL': str(url[:60] + '...' if len(url) > 60 else url),
                    'PageRank': f"{score:.6f}",
                    'Section': str(section),
                    'Title': str(page_data.get('title', '')[:40] + '...' if len(page_data.get('title', '')) > 40 else page_data.get('title', ''))
                })
            
            top_pages_df = pd.DataFrame(top_pages_data)
            st.dataframe(top_pages_df, use_container_width=True)
            
            # Question 3
            st.markdown('<div class="question-header">3. Do these align with your Priority Target Pages?</div>', unsafe_allow_html=True)
            
            if priority_analysis:
                found_count = len([p for p in priority_analysis if p['found']])
                total_count = len(priority_analysis)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Coverage</h3>
                        <h2>{found_count}/{total_count}</h2>
                        <p>{(found_count/total_count*100):.1f}% found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if found_count > 0:
                        avg_pr = np.mean([p['pagerank'] for p in priority_analysis if p['found']])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Avg PageRank</h3>
                            <h2>{avg_pr:.6f}</h2>
                            <p>Average for found pages</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    missing_count = total_count - found_count
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚠️ Missing</h3>
                        <h2>{missing_count}</h2>
                        <p>Pages not found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Priority pages table
                priority_table_data = []
                for p in priority_analysis:
                    priority_table_data.append({
                        'URL': str(p['url'][:50] + '...' if len(p['url']) > 50 else p['url']),
                        'PageRank': str(f"{p['pagerank']:.6f}" if p['found'] else 'Not Found'),
                        'Section': str(p['section']),
                        'Keywords': str(p['keywords'][:30] + '...' if len(str(p['keywords'])) > 30 else p['keywords'])
                    })
                
                priority_table_df = pd.DataFrame(priority_table_data)
                st.dataframe(priority_table_df, use_container_width=True)
            else:
                st.info("Upload a CSV file to analyze priority page alignment")
            
            # Question 4
            st.markdown('<div class="question-header">4. How can we reduce PageRank to non-valuable sections?</div>', unsafe_allow_html=True)
            
            # Calculate waste
            low_value_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'low')
            waste_percentage = (low_value_pr / total_pr * 100) if total_pr > 0 else 0
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔴 PageRank Waste</h3>
                    <h2>{waste_percentage:.1f}%</h2>
                    <p>Going to low-value sections</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # FIXED: Now this method exists
                fig_business = visualizer.create_business_value_pie()
                st.plotly_chart(fig_business, use_container_width=True)
            
            # Question 5
            st.markdown('<div class="question-header">5. How can we redirect PageRank to priority pages?</div>', unsafe_allow_html=True)
            
            strategies = []
            
            # High-PR low-value pages
            low_value_high_pr = []
            for url, score in pagerank_scores.items():
                section = section_mapping.get(url, 'other')
                if section_analyzer.get_business_value(section) == 'low' and score > 0.001:
                    low_value_high_pr.append((url, score, section))
            
            if low_value_high_pr:
                strategies.append({
                    'title': 'Reduce Links to Low-Value High-PR Pages',
                    'description': f'Found {len(low_value_high_pr)} low-value pages with significant PageRank',
                    'impact': 'High'
                })
            
            # Section linking issues
            section_linking = defaultdict(lambda: defaultdict(int))
            for source, target in crawler.graph.edges():
                source_section = section_mapping.get(source, 'other')
                target_section = section_mapping.get(target, 'other')
                section_linking[source_section][target_section] += 1
            
            for source_section, targets in section_linking.items():
                total_links = sum(targets.values())
                low_value_links = sum(count for target_section, count in targets.items() 
                                    if section_analyzer.get_business_value(target_section) == 'low')
                
                if total_links > 10 and (low_value_links / total_links) > 0.3:
                    strategies.append({
                        'title': f'Optimize {source_section.title()} Section Linking',
                        'description': f'{(low_value_links/total_links*100):.1f}% of links go to low-value sections',
                        'impact': 'Medium'
                    })
            
            if strategies:
                for i, strategy in enumerate(strategies, 1):
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{i}. {strategy['title']} ({strategy['impact']} Impact)</h4>
                        <p>{strategy['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No major redistribution issues detected!")
        
        with tab2:
            st.markdown("## 📊 Standard Visualizations")
            
            # Section heatmap
            st.markdown("### 🔥 Section Linking Heatmap")
            fig_heatmap = visualizer.create_section_matrix_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Route depth analysis
            st.markdown("### 📊 Route Depth Analysis")
            fig_depth = visualizer.create_route_depth_chart()
            st.plotly_chart(fig_depth, use_container_width=True)
            
            # Section statistics table
            st.markdown("### 📋 Section Statistics")
            section_table_data = []
            for s in section_stats:
                section_table_data.append({
                    'Section': str(s['section']),
                    'Total PageRank': f"{s['total_pr']:.4f}",
                    'Pages': int(s['page_count']),
                    'Percentage': f"{s['percentage']:.1f}%",
                    'Business Value': str(s['business_value']),
                    'Avg PR/Page': f"{s['total_pr']/s['page_count']:.6f}"
                })
            
            section_table_df = pd.DataFrame(section_table_data)
            st.dataframe(section_table_df, use_container_width=True)
        
        with tab3:
            st.markdown("## 🕸️ Interactive Network Graph")
            
            st.markdown("""
            This network visualization shows the top 25 pages and their PageRank flow relationships:
            - **Node size** represents PageRank score (larger = higher PageRank)
            - **Node color** represents business value (🟢 High, 🟡 Medium, 🔴 Low)
            - **Lines** show internal link connections between pages
            """)
            
            # Create and display network graph
            fig_network = visualizer.create_network_graph()
            
            if fig_network.data:
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Network insights
                st.markdown("### 📊 Network Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    node_count = len([url for url in pagerank_scores.keys() if url in [d for d in crawler.graph.nodes()]][:25])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔗 Nodes Displayed</h3>
                        <h2>{node_count}</h2>
                        <p>Top pages in network</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    edge_count = len(list(crawler.graph.subgraph([url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:25]]).edges()))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>↔️ Connections</h3>
                        <h2>{edge_count}</h2>
                        <p>Internal links shown</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Calculate network density
                    subgraph = crawler.graph.subgraph([url for url, _ in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:25]])
                    if len(subgraph.nodes()) > 1:
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
            else:
                st.info("No network data available for visualization")
        
        with tab4:
            st.markdown("## 🤖 AI-Powered Recommendations")
            
            if openai_key:
                if st.button("🧠 Generate AI Recommendations", type="primary"):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        ai_recommendations = generate_ai_recommendations(
                            pagerank_scores, section_stats, priority_analysis, openai_key
                        )
                        
                        st.markdown(f"""
                        <div class="insight-card">
                            <h3>🤖 AI Strategic Analysis</h3>
                            <div style="white-space: pre-wrap; line-height: 1.6;">{ai_recommendations}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>🤖 AI Recommendations Available</h4>
                    <p>Add your OpenAI API key in the sidebar to get intelligent, personalized recommendations for your PageRank optimization strategy.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Automated insights
            st.markdown("### 💡 Automated Insights")
            
            insights = []
            
            if waste_percentage > 15:
                insights.append(("warning", f"⚠️ PageRank Waste Detected", f"{waste_percentage:.1f}% of PageRank flows to low-value sections"))
            
            high_value_pr = sum(s['total_pr'] for s in section_stats if s['business_value'] == 'high')
            efficiency = (high_value_pr / total_pr * 100) if total_pr > 0 else 0
            
            if efficiency > 70:
                insights.append(("success", "✅ Good PageRank Efficiency", f"{efficiency:.1f}% flows to high-value sections"))
            elif efficiency < 50:
                insights.append(("warning", "⚠️ Low PageRank Efficiency", f"Only {efficiency:.1f}% flows to high-value sections"))
            
            for card_type, title, description in insights:
                st.markdown(f"""
                <div class="{card_type}-card">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("## 📋 Export & Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Generate Excel Report", type="primary"):
                    with st.spinner("Creating comprehensive Excel report..."):
                        excel_data = generate_excel_report(
                            crawler, pagerank_scores, section_mapping, section_analyzer, priority_df
                        )
                        
                        if excel_data:
                            st.success("✅ Excel report generated successfully!")
                            
                            # Create filename with timestamp
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"pagerank_analysis_{timestamp}.xlsx"
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            # Show what's included in the report
                            st.markdown("""
                            **📋 Report includes:**
                            - Executive Summary
                            - Top 100 Pages Analysis
                            - Section Performance Breakdown
                            - Priority Pages Analysis (if uploaded)
                            - Automated Recommendations
                            """)
                        else:
                            st.error("❌ Failed to generate Excel report. Please try again.")
            
            with col2:
                if st.button("📄 Generate JSON Report"):
                    report_data = {
                        'analysis_summary': {
                            'website_url': website_url,
                            'total_pages': len(pagerank_scores),
                            'total_links': len(crawler.graph.edges()),
                            'analysis_date': datetime.now().isoformat(),
                            'waste_percentage': waste_percentage,
                            'efficiency_score': efficiency
                        },
                        'section_analysis': section_stats,
                        'top_pages': [
                            {'url': url, 'pagerank': float(score), 'section': section_mapping.get(url, 'other')}
                            for url, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:50]
                        ],
                        'priority_analysis': priority_analysis or []
                    }
                    
                    report_json = json.dumps(report_data, indent=2, default=str)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"pagerank_analysis_{timestamp}.json"
                    
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=report_json,
                        file_name=filename,
                        mime="application/json"
                    )
            
            # Summary metrics
            st.markdown("### 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📄 Pages</h3>
                    <h2>{len(pagerank_scores)}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔗 Links</h3>
                    <h2>{len(crawler.graph.edges())}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️ Sections</h3>
                    <h2>{len(set(section_mapping.values()))}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ Efficiency</h3>
                    <h2>{efficiency:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
