import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page settings
st.set_page_config(
    page_title="📚 BookVibe - Smart Recommender", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS for modern aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8eaed;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .subtitle {
        text-align: center;
        color: #a8b2d1;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 3rem;
        opacity: 0.9;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #a8b2d1;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Card styling */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        color: #e8eaed;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Warning and success messages */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #e8eaed;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-subtitle {
        color: #a8b2d1;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    
    /* Book recommendation styling */
    .book-item {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 0 12px 12px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .book-item:hover {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        transform: translateX(5px);
    }
    
    .book-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #e8eaed;
        margin: 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 2rem 0;
        opacity: 0.6;
    }
    
    /* Hide Streamlit branding */
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load data (placeholder - replace with actual data loading)
try:
    books = pd.read_csv("books.csv")
    ratings = pd.read_csv("ratings.csv")
except:
    # Create sample data for demonstration
    books = pd.DataFrame({
        'Book_ID': range(1, 11),
        'Title': ['The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice', 
                 'The Catcher in the Rye', 'Lord of the Flies', 'Jane Eyre', 'Wuthering Heights',
                 'The Hobbit', 'Fahrenheit 451'],
        'Author': ['F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen',
                  'J.D. Salinger', 'William Golding', 'Charlotte Brontë', 'Emily Brontë',
                  'J.R.R. Tolkien', 'Ray Bradbury'],
        'Genre': ['Fiction', 'Fiction', 'Dystopian', 'Romance', 'Fiction', 'Fiction',
                 'Romance', 'Romance', 'Fantasy', 'Science Fiction']
    })
    ratings = pd.DataFrame({
        'User_ID': [1, 1, 1, 2, 2, 3, 3, 3],
        'Book_ID': [1, 2, 3, 1, 4, 2, 5, 6],
        'Rating': [5, 4, 5, 3, 4, 5, 4, 3]
    })

# Preprocessing - Content Based
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

# Preprocessing - Collaborative Filtering
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
if len(user_item_matrix) > 1:
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Header
st.markdown('<h1 class="main-header">📚 BookVibe</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Discover your next favorite book with AI-powered recommendations</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Create tabs with enhanced styling
tab1, tab2, tab3 = st.tabs(["🎯 Content Explorer", "👥 Community Picks", "🚀 Smart Fusion"])

# ------------------- TAB 1: Content-Based -------------------
with tab1:
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎯 Content Explorer</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Find books similar to your favorites based on title, author, and genre</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        book_titles = books['Title'].tolist()
        selected = st.selectbox(
            "🔍 Choose a book you enjoyed",
            book_titles,
            key="cb_book",
            help="Select a book to find similar recommendations"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("✨ Find Similar", key="cb_recommend"):
            with st.spinner("🔮 Analyzing book features..."):
                index = books[books['Title'] == selected].index[0]
                similar_books = content_similarity[index].argsort()[::-1][1:4]
                
                st.markdown("### 📖 Books You Might Love:")
                for i, book_idx in enumerate(similar_books, 1):
                    book = books.iloc[book_idx]
                    st.markdown(f"""
                    <div class="book-item">
                        <div class="book-title">📚 {book['Title']}</div>
                        <div style="color: #a8b2d1; font-size: 0.9rem; margin-top: 0.3rem;">
                            ✍️ by {book['Author']} • 🏷️ {book['Genre']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- TAB 2: Collaborative Filtering -------------------
with tab2:
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">👥 Community Picks</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Discover books loved by readers with similar tastes</p>', unsafe_allow_html=True)
    
    if len(user_item_matrix) > 1:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_ids = user_item_matrix.index.tolist()
            selected_user = st.selectbox(
                "👤 Select your reader profile",
                user_ids,
                key="cf_user",
                help="Choose a user ID to get personalized recommendations"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🤝 Get Recommendations", key="cf_recommend"):
                with st.spinner("🔍 Finding readers like you..."):
                    try:
                        sim_users = user_similarity_df.loc[selected_user].sort_values(ascending=False)[1:4].index
                        sim_ratings = user_item_matrix.loc[sim_users].mean().sort_values(ascending=False)
                        
                        user_rated_books = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user] > 0].index
                        recommendations = sim_ratings.drop(user_rated_books, errors='ignore').head(3)
                        
                        if len(recommendations) > 0:
                            st.markdown("### 🌟 Recommended by Similar Readers:")
                            for i, book_id in enumerate(recommendations.index, 1):
                                try:
                                    book = books[books['Book_ID'] == book_id].iloc[0]
                                    rating = recommendations[book_id]
                                    st.markdown(f"""
                                    <div class="book-item">
                                        <div class="book-title">📚 {book['Title']}</div>
                                        <div style="color: #a8b2d1; font-size: 0.9rem; margin-top: 0.3rem;">
                                            ✍️ by {book['Author']} • ⭐ {rating:.1f} avg rating
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                except:
                                    continue
                        else:
                            st.info("🔍 No new recommendations found. Try rating more books!")
                    except Exception as e:
                        st.error(f"⚠️ Unable to generate recommendations: {str(e)}")
    else:
        st.info("📊 Not enough user data available for collaborative filtering.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- TAB 3: Hybrid Recommender -------------------
with tab3:
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🚀 Smart Fusion</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">The best of both worlds - combining content similarity and community preferences</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(user_item_matrix) > 1:
            user_ids = user_item_matrix.index.tolist()
            selected_user_h = st.selectbox(
                "👤 Your reader profile",
                user_ids,
                key="hy_user",
                help="Select your user ID"
            )
        else:
            st.info("👤 User-based features not available")
            selected_user_h = None
    
    with col2:
        selected_book = st.selectbox(
            "📖 A book you enjoyed",
            books['Title'].tolist(),
            key="hy_book",
            help="Pick a book as starting point"
        )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🎯 Generate Smart Picks", key="hy_recommend"):
            with st.spinner("🧠 AI is thinking..."):
                try:
                    book_id = books[books['Title'] == selected_book]['Book_ID'].values[0]
                    book_index = books[books['Book_ID'] == book_id].index[0]
                    
                    content_scores = content_similarity[book_index]
                    
                    if selected_user_h and len(user_item_matrix) > 1:
                        user_ratings = user_item_matrix.loc[selected_user_h]
                        aligned_ratings = user_ratings.reindex(books['Book_ID']).fillna(0).values
                        
                        if user_ratings.sum() == 0:
                            st.warning("🤔 This user hasn't rated any books yet. Using content-based recommendations.")
                            hybrid_score = content_scores
                        else:
                            # Normalize scores
                            content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
                            rating_norm = (aligned_ratings - aligned_ratings.min()) / (aligned_ratings.max() - aligned_ratings.min() + 1e-8)
                            hybrid_score = 0.6 * content_norm + 0.4 * rating_norm
                    else:
                        hybrid_score = content_scores
                    
                    top_indices = np.argsort(hybrid_score)[::-1]
                    recommended_indices = [i for i in top_indices if i != book_index][:3]
                    
                    st.markdown("### 🎊 Your Personalized Recommendations:")
                    for i, book_idx in enumerate(recommended_indices, 1):
                        book = books.iloc[book_idx]
                        score = hybrid_score[book_idx]
                        st.markdown(f"""
                        <div class="book-item">
                            <div class="book-title">🏆 {book['Title']}</div>
                            <div style="color: #a8b2d1; font-size: 0.9rem; margin-top: 0.3rem;">
                                ✍️ by {book['Author']} • 🏷️ {book['Genre']} • 🎯 {score:.1%} match
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"⚠️ Error generating smart recommendations: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #a8b2d1; font-size: 0.9rem; opacity: 0.7; padding: 2rem;">
    🌟 Happy Reading! Made with ❤️ and AI • BookVibe 2024
</div>
""", unsafe_allow_html=True)