# 🍿 Netflix Content Recommender

An AI-powered content recommendation system that helps you discover your next favorite Netflix show or movie using advanced machine learning algorithms.

## 📋 Overview

This project implements a content-based recommendation system using TF-IDF vectorization and cosine similarity to analyze Netflix titles and provide personalized recommendations based on content features like description, genre, cast, and director.

## ✨ Features

- **Smart Recommendations**: Get personalized movie/TV show recommendations based on content similarity
- **Advanced Filtering**: Filter by content type (Movies/TV Shows), genre, and rating
- **Similarity Scoring**: See how closely each recommendation matches your selected title
- **Watchlist Management**: Save interesting titles to your personal watchlist
- **Random Discovery**: Explore random content when you can't decide
- **Detailed Information**: View comprehensive details including release year, rating, duration, and description
- **Interactive UI**: Clean, Netflix-themed interface built with Streamlit

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning (TF-IDF Vectorization, Cosine Similarity)
- **JSON**: Watchlist data persistence

## 📊 Dataset

The project uses the Netflix Movies and TV Shows dataset (`netflix_titles.csv`) containing:
- Title information
- Content type (Movie/TV Show)
- Director and cast details
- Genre categories
- Descriptions
- Release year, rating, and duration

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prakruthis0803/Netflix-Content-Recommender.git
   cd Netflix-Content-Recommender
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Preprocess the data**
   ```bash
   python preprocessing.py
   ```
   This will create `processed_data.csv` in the data folder.

## 💻 Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

3. **Get Recommendations**
   - Select a movie or TV show from the dropdown
   - Adjust the number of recommendations (1-20)
   - Apply filters for type, genre, and rating
   - Click "Get Recommendations" to see similar content
   - Add favorites to your watchlist

## 📁 Project Structure

```
Netflix-Content-Recommender/
│
├── app.py                      # Main Streamlit application
├── preprocessing.py            # Data preprocessing script
├── requirements.txt            # Project dependencies
├── .gitignore                 # Git ignore file
│
└── data/
    ├── netflix_titles.csv     # Raw Netflix dataset
    └── processed_data.csv     # Processed data (generated)
```

## 🧠 How It Works

1. **Data Preprocessing**: Combines text features (description, genre, cast, director) into a unified "tags" column
2. **TF-IDF Vectorization**: Converts text data into numerical vectors
3. **Cosine Similarity**: Calculates similarity scores between content items
4. **Recommendation Engine**: Returns top-N most similar items based on cosine similarity scores
5. **Filtering**: Applies user-selected filters (type, genre, rating) to refine results

## 🎯 Features in Detail

### Recommendation Algorithm
- Uses content-based filtering approach
- Analyzes multiple features: description, genres, cast, director, and content type
- Calculates similarity using cosine similarity metric
- Provides similarity scores (0-100%) for transparency

### User Interface
- Netflix-inspired dark theme
- Responsive design for better user experience
- Real-time filtering and recommendation updates
- Persistent watchlist across sessions
- Export watchlist to JSON

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Prakruthi S**
- GitHub: [@prakruthis0803](https://github.com/prakruthis0803)

## 🙏 Acknowledgments

- Netflix for inspiration
- The open-source community for the dataset
- Streamlit for the amazing framework

---

⭐ If you find this project helpful, please consider giving it a star!
