from flask import Flask, render_template, request, jsonify, send_file
from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
import os
import warnings
import io
import base64
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('YOUTUBE_API_KEY')

class PoliticalSentimentAnalyzer:
    def __init__(self, api_key):
        """Initialize the analyzer with YouTube API key"""
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.setup_sentiment_model()
        
    def setup_sentiment_model(self):
        """Load the multilingual sentiment analysis model"""
        print("🔄 Loading sentiment analysis model...")
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        print("✅ Model loaded successfully!")
    
    def get_sentiment(self, text):
        """Analyze sentiment of text"""
        try:
            # Clean text
            text = re.sub(r'[^\w\s]', '', text)
            text = text[:512]  # Limit to 512 characters
            
            result = self.sentiment_pipeline(text)[0]
            return {
                "label": result['label'].replace('LABEL_', '').capitalize(),
                "score": result['score']
            }
        except Exception as e:
            return {"label": "Neutral", "score": 0.5}
    
    def fetch_comments(self, query, party_name, num_videos=5, comments_per_video=50):
        """Fetch comments from YouTube videos"""
        print(f"\n🔍 Searching YouTube for: {query}")
        comments = []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=num_videos,
                order='relevance'
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            print(f"📹 Found {len(video_ids)} videos")
            
            for i, video_id in enumerate(video_ids, 1):
                try:
                    print(f"   Processing video {i}/{len(video_ids)}...")
                    response = self.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=comments_per_video,
                        textFormat='plainText',
                        order='relevance'
                    ).execute()
                    
                    for item in response['items']:
                        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                        comments.append({
                            "party": party_name,
                            "comment": comment,
                            "video_id": video_id
                        })
                        
                except Exception as e:
                    print(f"   ⚠️ Error fetching comments from video {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error with search query: {e}")
            
        print(f"📥 Fetched {len(comments)} comments for {party_name}")
        return comments
    
    def analyze_comments(self, comments):
        """Analyze sentiment of all comments"""
        print("\n🔄 Analyzing sentiment...")
        analyzed = []
        
        for i, item in enumerate(comments, 1):
            if i % 50 == 0:
                print(f"   Processed {i}/{len(comments)} comments...")
                
            sentiment = self.get_sentiment(item['comment'])
            analyzed.append({
                "party": item["party"],
                "comment": item["comment"],
                "sentiment": sentiment['label'],
                "confidence": sentiment['score'],
                "video_id": item["video_id"]
            })
            
        print(f"✅ Analyzed {len(analyzed)} comments")
        return analyzed
    
    def generate_summary_report(self, df, config):
        """Generate comprehensive summary report"""
        report = []
        report.append("="*80)
        report.append("📊 POLITICAL SENTIMENT ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Purpose: {config['purpose'].title()}")
        report.append(f"🏛️ Parties Analyzed: {' vs '.join(config['parties'])}")
        report.append(f"📈 Total Comments: {len(df)}")
        report.append("")
        
        # Overall sentiment distribution
        report.append("📊 OVERALL SENTIMENT DISTRIBUTION")
        report.append("-" * 50)
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"{sentiment}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Party-wise analysis
        for party in config['parties']:
            party_data = df[df['party'] == party]
            if len(party_data) == 0:
                continue
                
            report.append(f"🏛️ {party.upper()} ANALYSIS")
            report.append("-" * 50)
            report.append(f"📝 Total Comments: {len(party_data)}")
            
            # Sentiment breakdown
            party_sentiment = party_data['sentiment'].value_counts()
            report.append("📊 Sentiment Breakdown:")
            for sentiment, count in party_sentiment.items():
                percentage = (count / len(party_data)) * 100
                report.append(f"   {sentiment}: {count} ({percentage:.1f}%)")
            
            # Average confidence
            avg_confidence = party_data['confidence'].mean()
            report.append(f"🎯 Average Confidence: {avg_confidence:.3f}")
            
            # Key insights
            report.append("💡 Key Insights:")
            positive_ratio = len(party_data[party_data['sentiment'] == 'Positive']) / len(party_data)
            negative_ratio = len(party_data[party_data['sentiment'] == 'Negative']) / len(party_data)
            
            if positive_ratio > 0.5:
                report.append(f"   ✅ Generally positive public perception ({positive_ratio:.1%})")
            elif negative_ratio > 0.5:
                report.append(f"   ❌ Concerning negative sentiment ({negative_ratio:.1%})")
            else:
                report.append(f"   ⚖️ Mixed public sentiment")
            
            # Strengths and weaknesses
            report.append("\n🔍 STRENGTHS:")
            if positive_ratio > 0.3:
                report.append("   • Strong positive engagement from supporters")
            if avg_confidence > 0.7:
                report.append("   • High confidence in sentiment predictions")
            if len(party_data) > len(df) * 0.4:
                report.append("   • High online discussion volume")
            
            report.append("\n⚠️ AREAS FOR IMPROVEMENT:")
            if negative_ratio > 0.3:
                report.append("   • Address negative sentiment concerns")
            if negative_ratio > positive_ratio:
                report.append("   • Focus on positive messaging and communication")
            if avg_confidence < 0.6:
                report.append("   • Mixed public opinion - need clearer positioning")
            
            report.append("\n📋 RECOMMENDATIONS:")
            if config['purpose'] == 'manifesto development':
                if negative_ratio > 0.4:
                    report.append("   • Include policies addressing public concerns")
                    report.append("   • Focus on transparency and accountability")
                if positive_ratio > 0.4:
                    report.append("   • Leverage popular policies in manifesto")
                    report.append("   • Highlight successful initiatives")
            elif config['purpose'] == 'election prediction':
                if positive_ratio > negative_ratio:
                    report.append("   • Current sentiment favors this party")
                    report.append("   • Maintain positive momentum")
                else:
                    report.append("   • Need strategic communication improvement")
                    report.append("   • Address voter concerns proactively")
            
            report.append("\n" + "="*50)
        
        # Comparative analysis
        if len(config['parties']) == 2:
            report.append("\n🔄 COMPARATIVE ANALYSIS")
            report.append("-" * 50)
            
            party1_data = df[df['party'] == config['parties'][0]]
            party2_data = df[df['party'] == config['parties'][1]]
            
            if len(party1_data) > 0 and len(party2_data) > 0:
                p1_positive = len(party1_data[party1_data['sentiment'] == 'Positive']) / len(party1_data)
                p2_positive = len(party2_data[party2_data['sentiment'] == 'Positive']) / len(party2_data)
                
                if p1_positive > p2_positive:
                    leader = config['parties'][0]
                    difference = (p1_positive - p2_positive) * 100
                else:
                    leader = config['parties'][1]
                    difference = (p2_positive - p1_positive) * 100
                
                report.append(f"🏆 {leader} has {difference:.1f}% higher positive sentiment")
                
                # Engagement comparison
                if len(party1_data) > len(party2_data):
                    report.append(f"📢 {config['parties'][0]} has higher online engagement")
                else:
                    report.append(f"📢 {config['parties'][1]} has higher online engagement")
        
        return "\n".join(report)
    
    def create_visualizations(self, df, config):
        """Create visualization plots and return as base64 encoded images"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Political Sentiment Analysis: {' vs '.join(config['parties'])}", fontsize=16, fontweight='bold')
        
        # 1. Overall sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Party-wise sentiment comparison
        sns.countplot(data=df, x="sentiment", hue="party", ax=axes[0, 1], palette="Set2")
        axes[0, 1].set_title('Sentiment Comparison by Party')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Number of Comments')
        
        # 3. Confidence distribution
        df['confidence'].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Confidence Score Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Party-wise positive sentiment ratio
        party_stats = df.groupby('party')['sentiment'].apply(
            lambda x: (x == 'Positive').sum() / len(x) * 100
        ).sort_values(ascending=True)
        
        bars = axes[1, 1].barh(party_stats.index, party_stats.values, color=['#3498db', '#e74c3c'])
        axes[1, 1].set_title('Positive Sentiment Ratio by Party')
        axes[1, 1].set_xlabel('Positive Sentiment (%)')
        
        # Add value labels on bars
        for i, v in enumerate(party_stats.values):
            axes[1, 1].text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        
        # Convert plot to base64 encoded image
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        party1 = request.form.get('party1', '').strip()
        party2 = request.form.get('party2', '').strip()
        purpose = request.form.get('purpose', 'general political analysis').strip()
        query1 = request.form.get('query1', f"{party1} political speeches latest").strip()
        query2 = request.form.get('query2', f"{party2} political speeches latest").strip()
        num_videos = int(request.form.get('num_videos', 5))
        comments_per_video = int(request.form.get('comments_per_video', 50))
        
        config = {
            "parties": [party1, party2],
            "queries": [query1, query2],
            "purpose": purpose,
            "num_videos": num_videos,
            "comments_per_video": comments_per_video
        }
        
        # Initialize analyzer
        analyzer = PoliticalSentimentAnalyzer(API_KEY)
        
        # Fetch and analyze comments
        all_comments = []
        for party, query in zip(config['parties'], config['queries']):
            comments = analyzer.fetch_comments(query, party, config['num_videos'], config['comments_per_video'])
            all_comments.extend(comments)
        
        if not all_comments:
            return render_template('index.html', error="No comments found. Please check your search queries and try again.")
        
        analyzed_comments = analyzer.analyze_comments(all_comments)
        df = pd.DataFrame(analyzed_comments)
        
        # Generate report and visualizations
        report = analyzer.generate_summary_report(df, config)
        plot_url = analyzer.create_visualizations(df, config)
        
        # Prepare data for template
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        party_sentiments = {}
        
        for party in config['parties']:
            party_data = df[df['party'] == party]
            if len(party_data) > 0:
                party_sentiments[party] = {
                    'count': len(party_data),
                    'sentiments': party_data['sentiment'].value_counts().to_dict(),
                    'avg_confidence': party_data['confidence'].mean()
                }
        
        return render_template('index.html', 
                             report=report,
                             plot_url=plot_url,
                             sentiment_counts=sentiment_counts,
                             party_sentiments=party_sentiments,
                             config=config)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)