import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json

# 1. Page Configuration
st.set_page_config(
    page_title="Netflix AI Recommender", 
    page_icon="🍿",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .recommendation-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e50914;
        margin: 10px 0;
    }
    .similarity-score {
        background-color: #e50914;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .metric-container {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .watchlist-item {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #e50914;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for watchlist and recommendations
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []
if 'current_recommendations' not in st.session_state:
    st.session_state['current_recommendations'] = None
if 'show_recommendations' not in st.session_state:
    st.session_state['show_recommendations'] = False

# 2. Function to Load and Preprocess Data
@st.cache_data
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('data/netflix_titles.csv')
    
    # Select columns and handle missing values
    df = df[['title', 'type', 'director', 'cast', 'listed_in', 'description', 'release_year', 'rating', 'duration']].fillna('')
    
    # Create the 'tags' column for the AI to analyze
    df['tags'] = (df['description'] + " " + 
                  df['listed_in'] + " " + 
                  df['cast'] + " " + 
                  df['director'] + " " + 
                  df['type']).apply(lambda x: x.lower())
    
    return df

# 3. Cache similarity matrix for performance
@st.cache_data
def compute_similarity_matrix(_data):
    # Use TF-IDF for better word importance weighting
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    vectors = tfidf.fit_transform(_data['tags']).toarray()
    
    # Calculate similarity between all items
    similarity = cosine_similarity(vectors)
    return similarity

# 4. Enhanced Recommendation Engine
def get_recommendations(target_title, data, similarity, n_recommendations=5, content_filter=None, year_range=None):
    # Apply filters if specified
    filtered_data = data.copy()
    
    if content_filter and content_filter != "All":
        filtered_data = filtered_data[filtered_data['type'] == content_filter]
    
    if year_range:
        filtered_data = filtered_data[
            (filtered_data['release_year'] >= year_range[0]) & 
            (filtered_data['release_year'] <= year_range[1])
        ]
    
    filtered_data = filtered_data.reset_index(drop=True)
    
    # Recompute similarity for filtered data
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    vectors = tfidf.fit_transform(filtered_data['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    # Find the index of the selected content
    idx = filtered_data[filtered_data['title'] == target_title].index[0]
    
    # Get top N matches with similarity scores
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    recommendations = []
    for i in distances[1:n_recommendations+1]:
        rec_idx = i[0]
        recommendations.append({
            'title': filtered_data.iloc[rec_idx]['title'],
            'type': filtered_data.iloc[rec_idx]['type'],
            'genres': filtered_data.iloc[rec_idx]['listed_in'],
            'description': filtered_data.iloc[rec_idx]['description'],
            'year': filtered_data.iloc[rec_idx]['release_year'],
            'rating': filtered_data.iloc[rec_idx]['rating'],
            'duration': filtered_data.iloc[rec_idx]['duration'],
            'similarity': round(i[1] * 100, 1)
        })
    
    return recommendations

# 5. NEW: Get recommendations based on multiple titles (watchlist)
def get_watchlist_recommendations(watchlist_titles, data, n_recommendations=10, content_filter=None, year_range=None):
    # Apply filters if specified
    filtered_data = data.copy()
    
    if content_filter and content_filter != "All":
        filtered_data = filtered_data[filtered_data['type'] == content_filter]
    
    if year_range:
        filtered_data = filtered_data[
            (filtered_data['release_year'] >= year_range[0]) & 
            (filtered_data['release_year'] <= year_range[1])
        ]
    
    filtered_data = filtered_data.reset_index(drop=True)
    
    # Recompute similarity for filtered data
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    vectors = tfidf.fit_transform(filtered_data['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    # Aggregate similarity scores from all watchlist items
    aggregated_scores = {}
    
    for title in watchlist_titles:
        if title in filtered_data['title'].values:
            idx = filtered_data[filtered_data['title'] == title].index[0]
            distances = list(enumerate(similarity[idx]))
            
            for i, score in distances:
                movie_title = filtered_data.iloc[i]['title']
                if movie_title not in watchlist_titles:  # Exclude watchlist items
                    if movie_title in aggregated_scores:
                        aggregated_scores[movie_title] += score
                    else:
                        aggregated_scores[movie_title] = score
    
    # Average the scores
    for title in aggregated_scores:
        aggregated_scores[title] /= len(watchlist_titles)
    
    # Sort by score and get top N
    top_recommendations = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    recommendations = []
    for title, score in top_recommendations:
        movie_data = filtered_data[filtered_data['title'] == title].iloc[0]
        recommendations.append({
            'title': movie_data['title'],
            'type': movie_data['type'],
            'genres': movie_data['listed_in'],
            'description': movie_data['description'],
            'year': movie_data['release_year'],
            'rating': movie_data['rating'],
            'duration': movie_data['duration'],
            'similarity': round(score * 100, 1)
        })
    
    return recommendations

# --- USER INTERFACE ---
st.title("🍿 Netflix Content Recommender")
st.markdown("### Powered by Advanced NLP & Machine Learning")
st.info("🎯 This system uses TF-IDF vectorization and cosine similarity to suggest content based on plot, cast, genre, and more!")

try:
    df = load_and_clean_data()
    
    # Compute similarity matrix once
    similarity_matrix = compute_similarity_matrix(df)
    
    # Sidebar for filters and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Content type filter
        content_filter = st.selectbox(
            "Filter by Type:",
            ["All", "Movie", "TV Show"]
        )
        
        # Year range filter
        min_year = int(df['release_year'].min())
        max_year = int(df['release_year'].max())
        
        year_range = st.slider(
            "Filter by Year Range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
        
        # Number of recommendations
        n_recs = st.slider(
            "Number of Recommendations:",
            min_value=3,
            max_value=15,
            value=5
        )
        
        # Random discovery button
        st.markdown("---")
        st.subheader("🎲 Feeling Lucky?")
        if st.button("Surprise Me!", use_container_width=True):
            # Apply filters to random selection
            temp_df = df.copy()
            if content_filter != "All":
                temp_df = temp_df[temp_df['type'] == content_filter]
            temp_df = temp_df[
                (temp_df['release_year'] >= year_range[0]) & 
                (temp_df['release_year'] <= year_range[1])
            ]
            if len(temp_df) > 0:
                random_title = random.choice(temp_df['title'].values)
                st.session_state['selected_title'] = random_title
        
        # NEW: Watchlist Section
        st.markdown("---")
        st.subheader("⭐ My Watchlist")
        
        if len(st.session_state['watchlist']) > 0:
            st.write(f"**{len(st.session_state['watchlist'])} items**")
            
            # Display watchlist items
            for item in st.session_state['watchlist']:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="watchlist-item">
                        {item}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("❌", key=f"remove_{item}"):
                        st.session_state['watchlist'].remove(item)
                        st.rerun()
            
            # Get recommendations based on watchlist
            if st.button("🎬 Get Watchlist Recommendations", use_container_width=True, type="primary"):
                st.session_state['show_watchlist_recs'] = True
            
            # Clear watchlist button
            if st.button("🗑️ Clear Watchlist", use_container_width=True):
                st.session_state['watchlist'] = []
                st.session_state['show_watchlist_recs'] = False
                st.rerun()
        else:
            st.info("Add titles to your watchlist to get personalized recommendations!")
        
        # Display stats
        st.markdown("---")
        st.subheader("📊 Dataset Stats")
        st.metric("Total Content", len(df))
        st.metric("Movies", len(df[df['type'] == 'Movie']))
        st.metric("TV Shows", len(df[df['type'] == 'TV Show']))
    
    # Filter titles based on content type and year
    filtered_df = df.copy()
    if content_filter != "All":
        filtered_df = filtered_df[filtered_df['type'] == content_filter]
    
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= year_range[0]) & 
        (filtered_df['release_year'] <= year_range[1])
    ]
    
    titles = filtered_df['title'].values
    
    # Main content selection
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        # Use session state for random selection
        if 'selected_title' not in st.session_state:
            st.session_state['selected_title'] = titles[0] if len(titles) > 0 else df['title'].values[0]
        
        # Ensure selected title is in filtered titles
        if st.session_state['selected_title'] not in titles:
            st.session_state['selected_title'] = titles[0] if len(titles) > 0 else df['title'].values[0]
        
        selected_movie = st.selectbox(
            "Type or select a Netflix title:",
            titles,
            index=list(titles).index(st.session_state['selected_title']) if st.session_state['selected_title'] in titles else 0
        )
        st.session_state['selected_title'] = selected_movie
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_btn = st.button('🔍 Get Recommendations', type="primary", use_container_width=True)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        # Add to watchlist button
        if selected_movie not in st.session_state['watchlist']:
            if st.button('⭐ Add to Watchlist', use_container_width=True, key='main_add_watchlist'):
                st.session_state['watchlist'].append(selected_movie)
                st.toast(f"✅ Added '{selected_movie}' to watchlist!", icon="⭐")
        else:
            st.button('✅ In Watchlist', use_container_width=True, disabled=True)
    
    # Display selected content info
    selected_info = df[df['title'] == selected_movie].iloc[0]
    with st.expander("ℹ️ About Selected Content", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Type:** {selected_info['type']}")
            st.write(f"**Year:** {selected_info['release_year']}")
        with col2:
            st.write(f"**Rating:** {selected_info['rating']}")
            st.write(f"**Duration:** {selected_info['duration']}")
        with col3:
            st.write(f"**Genres:** {selected_info['listed_in']}")
        st.write(f"**Description:** {selected_info['description']}")

    # Show watchlist recommendations
    if 'show_watchlist_recs' in st.session_state and st.session_state['show_watchlist_recs']:
        if len(st.session_state['watchlist']) > 0:
            with st.spinner('🤖 Analyzing your watchlist with AI...'):
                watchlist_recs = get_watchlist_recommendations(
                    st.session_state['watchlist'],
                    df,
                    n_recommendations=n_recs * 2,
                    content_filter=content_filter if content_filter != "All" else None,
                    year_range=year_range
                )
                
                st.markdown(f"## 🎬 Based on Your Watchlist ({len(st.session_state['watchlist'])} titles)")
                st.markdown("---")
                
                # Display recommendations in cards
                for i, rec in enumerate(watchlist_recs, 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>#{i} {rec['title']} <span class="similarity-score">{rec['similarity']}% Match</span></h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2.5, 0.8])
                    with col1:
                        st.write(f"**{rec['type']}**")
                    with col2:
                        st.write(f"📅 {rec['year']}")
                    with col3:
                        st.write(f"⭐ {rec['rating']}")
                    with col4:
                        st.write(f"⏱️ {rec['duration']}")
                    with col5:
                        st.write(f"🎭 {rec['genres']}")
                    with col6:
                        # Add to watchlist button for each recommendation
                        if rec['title'] not in st.session_state['watchlist']:
                            if st.button('➕', key=f"add_watchlist_rec_{i}", help="Add to Watchlist"):
                                st.session_state['watchlist'].append(rec['title'])
                                st.toast(f"✅ Added '{rec['title']}' to watchlist!", icon="⭐")
                        else:
                            st.button('✅', key=f"in_watchlist_rec_{i}", disabled=True, help="Already in Watchlist")
                    
                    st.write(f"**Plot:** {rec['description']}")
                    st.markdown("---")
        
        st.session_state['show_watchlist_recs'] = False

    # Handle recommendation button click
    if recommend_btn:
        st.session_state['show_recommendations'] = True
        st.session_state['current_recommendations'] = {
            'title': selected_movie,
            'recs': None
        }

    # Display regular recommendations if triggered
    if st.session_state['show_recommendations'] and st.session_state['current_recommendations']:
        # Generate recommendations if not already cached
        if st.session_state['current_recommendations']['recs'] is None:
            with st.spinner('🤖 Analyzing patterns with AI...'):
                recommendations = get_recommendations(
                    selected_movie, 
                    df, 
                    similarity_matrix, 
                    n_recommendations=n_recs,
                    content_filter=content_filter if content_filter != "All" else None,
                    year_range=year_range
                )
                st.session_state['current_recommendations']['recs'] = recommendations
        else:
            recommendations = st.session_state['current_recommendations']['recs']
        
        st.markdown(f"## 🎬 Because you liked **{st.session_state['current_recommendations']['title']}**")
        st.markdown("---")
        
        # Display recommendations in cards
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>#{i} {rec['title']} <span class="similarity-score">{rec['similarity']}% Match</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2.5, 0.8])
            with col1:
                st.write(f"**{rec['type']}**")
            with col2:
                st.write(f"📅 {rec['year']}")
            with col3:
                st.write(f"⭐ {rec['rating']}")
            with col4:
                st.write(f"⏱️ {rec['duration']}")
            with col5:
                st.write(f"🎭 {rec['genres']}")
            with col6:
                # Add to watchlist button for each recommendation
                if rec['title'] not in st.session_state['watchlist']:
                    if st.button('➕', key=f"add_rec_{i}", help="Add to Watchlist"):
                        st.session_state['watchlist'].append(rec['title'])
                        st.toast(f"✅ Added '{rec['title']}' to watchlist!", icon="⭐")
                else:
                    st.button('✅', key=f"already_rec_{i}", disabled=True, help="Already in Watchlist")
            
            st.write(f"**Plot:** {rec['description']}")
            st.markdown("---")

except Exception as e:
    st.error(f"Error: {e}. Please ensure 'netflix_titles.csv' is in the 'data' folder.")