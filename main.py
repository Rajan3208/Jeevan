import streamlit as st

st.set_page_config(
    page_title="Document Insights Extractor",
    page_icon="📄",
    layout="wide"
)

import os
import base64
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
import warnings
import time

warnings.filterwarnings("ignore")

@st.cache_resource
def load_nlp_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

def convert_pdf_to_images(file_bytes, scale=300/72, max_pages=5):
    import pypdfium2 as pdfium
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    list_final_images = []
    try:
        pdf_file = pdfium.PdfDocument(tmp_path)  
        page_indices = [i for i in range(min(len(pdf_file), max_pages))]
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices, 
            scale=scale,
        )
        for i, image in zip(page_indices, renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='jpeg', optimize=True, quality=85)
            list_final_images.append(dict({i: image_byte_array.getvalue()}))
        pdf_file.close()
        del renderer
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    return list_final_images

def extract_text_with_pyPDF(file_bytes):
    from PyPDF2 import PdfReader
    with BytesIO(file_bytes) as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        max_pages = min(len(pdf_reader.pages), 10)
        raw_text = ''
        for i in range(max_pages):
            text = pdf_reader.pages[i].extract_text()
            if text:
                raw_text += text
        return raw_text

@st.cache_data
def extract_insights_with_spacy(text):
    nlp = load_nlp_model()
    doc = nlp(text[:20000])
    entities = {}
    for ent in doc.ents:
        if len(entities) >= 50:
            break
        entities[ent.text] = ent.label_
    noun_chunks = [chunk.text for chunk in doc.noun_chunks][:20]
    sentences = list(doc.sents)
    important_sentences = sorted(
        sentences, 
        key=lambda s: sum(1 for ent in doc.ents if ent.start >= s.start and ent.end <= s.end), 
        reverse=True
    )[:3]
    summary = ' '.join([sent.text for sent in important_sentences])
    insights = {
        'entities': entities,
        'key_phrases': noun_chunks[:10],
        'summary': summary
    }
    return insights, f"Document contains {len(entities)} named entities including {', '.join(list(entities.keys())[:5])}. Key topics focus on {', '.join(noun_chunks[:3])}."

@st.cache_data
def extract_insights_with_sklearn(text, num_topics=2):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2, 
        max_features=5000,
        stop_words='english'
    )
    try:
        dtm = vectorizer.fit_transform([text])
        lda_model = LatentDirichletAllocation(
            n_components=num_topics, 
            random_state=42,
            max_iter=5
        )
        lda_model.fit(dtm)
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
        model_filename = 'topic_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump((lda_model, vectorizer), f)
        topic_insights = f"Main topics: {', '.join([' & '.join(topic[:3]) for topic in topics])}"
        return topics, topic_insights, model_filename
    except Exception as e:
        st.warning(f"Topic modeling error: {e}")
        return [], "Could not extract topics due to insufficient text data", None

def simple_sentiment_analysis(text):
    positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'pleased', 'satisfied', 'wonderful', 'success']
    negative_words = ['bad', 'poor', 'terrible', 'negative', 'sad', 'unhappy', 'disappointed', 'awful', 'failure']
    text_lower = text.lower()
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE" if negative_count > positive_count else "NEUTRAL"
    sentences = text.split('.')
    summary_text = '. '.join(sentences[:5]) + '.' if len(sentences) > 5 else text
    return {
        'sentiment': sentiment,
        'summary': summary_text
    }, f"Document sentiment: {sentiment}. {summary_text[:100]}..."

def generate_comprehensive_insights(text, file_name):
    spacy_insights, spacy_summary = extract_insights_with_spacy(text)
    topics, topic_summary, topic_model_file = extract_insights_with_sklearn(text)
    transformer_results, transformer_summary = simple_sentiment_analysis(text)
    langchain_summary = '. '.join(text.split('.')[:5]) + '.'
    langchain_insights = f"Key insights: {langchain_summary[:150]}..."
    insights_summary = f"""
    DOCUMENT INSIGHTS SUMMARY:
    
    {spacy_summary}
    
    {topic_summary}
    
    {transformer_summary}
    
    {langchain_insights}
    """
    detailed_insights = {
        'file_name': file_name,
        'spacy': spacy_insights,
        'topics': topics,
        'transformer': transformer_results,
        'langchain': langchain_summary,
        'saved_models': {
            'topic_model': topic_model_file,
            'keras_model': None
        }
    }
    with open('document_insights.pkl', 'wb') as f:
        pickle.dump(detailed_insights, f)
    return insights_summary.strip(), detailed_insights

def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Extracting text from PDF...")
    pypdf_text = extract_text_with_pyPDF(file_bytes)
    progress_bar.progress(50)
    combined_text = pypdf_text
    status_text.text("Generating insights...")
    insights, detailed_insights = generate_comprehensive_insights(combined_text, uploaded_file.name)
    progress_bar.progress(100)
    processing_time = time.time() - start_time
    status_text.text(f"Processing complete in {processing_time:.2f} seconds!")
    return {
        'file_name': uploaded_file.name,
        'text': {
            'pypdf': pypdf_text,
        },
        'images': [],
        'insights': insights,
        'detailed_insights': detailed_insights,
        'processing_time': processing_time
    }

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'

def main():
    st.title("📄 Fast Document Insights Extractor (No OCR)")
    st.markdown("""
    Upload a PDF document to extract text and generate comprehensive insights.
    
    This version:
    - Uses PyPDF2 only (no OCR fallback)
    - spaCy for fast entity recognition
    - Lightweight topic modeling
    - Rule-based sentiment analysis
    """)
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        st.markdown(f"### Processing: {uploaded_file.name}")
        results = process_pdf(uploaded_file)
        st.success(f"Processing completed in {results['processing_time']:.2f} seconds!")
        tab1, tab2, tab3 = st.tabs(["📊 Insights", "📝 Extracted Text", "💾 Models"])
        
        with tab1:
            st.markdown("### Document Insights")
            st.markdown(results['insights'])
            st.subheader("Named Entities")
            entities = results['detailed_insights']['spacy']['entities']
            if entities:
                import pandas as pd
                st.dataframe(pd.DataFrame(list(entities.items()), columns=['Entity', 'Type']))
            st.subheader("Topics")
            topics = results['detailed_insights']['topics']
            if topics:
                for i, topic in enumerate(topics):
                    st.markdown(f"**Topic {i+1}:** {', '.join(topic[:10])}")
            st.subheader("Sentiment")
            st.markdown(f"**Sentiment:** {results['detailed_insights']['transformer']['sentiment']}")
        
        with tab2:
            st.markdown("### Extracted Text")
            st.text_area("Text extracted with PyPDF2", results['text']['pypdf'], height=300)
        
        with tab3:
            st.markdown("### Download Models")
            if os.path.exists('topic_model.pkl'):
                st.markdown(get_binary_file_downloader_html('topic_model.pkl', 'Download Topic Model'), unsafe_allow_html=True)
            if os.path.exists('document_insights.pkl'):
                st.markdown(get_binary_file_downloader_html('document_insights.pkl', 'Download Insights'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
