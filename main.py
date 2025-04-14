import streamlit as st

# IMPORTANT: This must be the first Streamlit command
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

# Now import the other libraries - only import what's needed when needed
@st.cache_resource
def load_nlp_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Safe PDF handling
def convert_pdf_to_images(file_bytes, scale=300/72, max_pages=5):
    """Convert PDF binary data to list of images - limited to first 5 pages for speed"""
    import pypdfium2 as pdfium
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    list_final_images = []
    
    try:
        pdf_file = pdfium.PdfDocument(tmp_path)  
        # Limit to max_pages for speed
        page_indices = [i for i in range(min(len(pdf_file), max_pages))]
        
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices, 
            scale=scale,
        )
        
        for i, image in zip(page_indices, renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='jpeg', optimize=True, quality=85)
            image_byte_array = image_byte_array.getvalue()
            list_final_images.append(dict({i:image_byte_array}))
        
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

# Extract text from PDF using PyPDF2 - fastest method
def extract_text_with_pyPDF(file_bytes):
    from PyPDF2 import PdfReader
    
    with BytesIO(file_bytes) as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        
        # Limit to first 10 pages for speed
        max_pages = min(len(pdf_reader.pages), 10)
        raw_text = ''
        for i in range(max_pages):
            text = pdf_reader.pages[i].extract_text()
            if text:
                raw_text += text

        return raw_text

# Extract text with PyTesseract - only if PyPDF2 fails to get good content
def extract_text_with_pytesseract(list_dict_final_images):
    """Extract text from images using PyTesseract"""
    # Only process if needed
    import pytesseract
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        pass
        
    # Only process first 3 images max for speed
    image_limit = min(len(list_dict_final_images), 3)
    image_list = [list(data.values())[0] for data in list_dict_final_images[:image_limit]]
    image_content = []
    
    for image_bytes in image_list:
        image = Image.open(BytesIO(image_bytes))
        raw_text = pytesseract.image_to_string(image)
        image_content.append(raw_text)
    
    return "\n".join(image_content)

# Extract insights using spaCy - fastest NLP method
@st.cache_data
def extract_insights_with_spacy(text):
    """Extract key entities and insights using spaCy"""
    nlp = load_nlp_model()
    
    # Limit text size for speed
    max_text_len = min(len(text), 20000)
    doc = nlp(text[:max_text_len])
    
    # Extract named entities
    entities = {}
    for ent in doc.ents:
        # Only store first 50 entities max
        if len(entities) >= 50:
            break
        entities[ent.text] = ent.label_
    
    # Extract key phrases (noun chunks) - limited to 20
    noun_chunks = [chunk.text for chunk in doc.noun_chunks][:20]
    
    # Extract main sentences based on entity density
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

# Extract insights using topic modeling - simplified for speed
@st.cache_data
def extract_insights_with_sklearn(text, num_topics=2):
    """Extract topics using NMF topic modeling"""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    # Create vectorizer - using simpler CountVectorizer instead of TF-IDF for speed
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2, 
        max_features=5000,  # Limit features
        stop_words='english'
    )
    
    # Create document-term matrix
    try:
        dtm = vectorizer.fit_transform([text])
    
        # Use LDA instead of NMF - faster
        lda_model = LatentDirichletAllocation(
            n_components=num_topics, 
            random_state=42,
            max_iter=5  # Reduced iterations for speed
        )
        lda_model.fit(dtm)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)
            
        # Save topic model - quick pickle
        model_filename = 'topic_model.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump((lda_model, vectorizer), f)
            
        topic_insights = f"Main topics: {', '.join([' & '.join(topic[:3]) for topic in topics])}"
        return topics, topic_insights, model_filename
    except Exception as e:
        st.warning(f"Topic modeling error: {e}")
        return [], "Could not extract topics due to insufficient text data", None

# Very simple sentiment analysis - no heavy models
def simple_sentiment_analysis(text):
    positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'pleased', 'satisfied', 'wonderful', 'success']
    negative_words = ['bad', 'poor', 'terrible', 'negative', 'sad', 'unhappy', 'disappointed', 'awful', 'failure']
    
    text_lower = text.lower()
    positive_count = sum(text_lower.count(word) for word in positive_words)
    negative_count = sum(text_lower.count(word) for word in negative_words)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    # Simple summarization - first few sentences
    sentences = text.split('.')
    summary_text = '. '.join(sentences[:5]) + '.' if len(sentences) > 5 else text
    
    return {
        'sentiment': sentiment,
        'summary': summary_text
    }, f"Document sentiment: {sentiment}. {summary_text[:100]}..."

# Generate comprehensive insights - simplified
def generate_comprehensive_insights(text, file_name):
    """Generate comprehensive insights using faster NLP approaches"""
    
    # SpaCy insights - fastest
    spacy_insights, spacy_summary = extract_insights_with_spacy(text)
    
    # Topic modeling insights - faster implementation
    topics, topic_summary, topic_model_file = extract_insights_with_sklearn(text)
    
    # Simple sentiment analysis - no heavy models
    transformer_results, transformer_summary = simple_sentiment_analysis(text)
    keras_model_file = None
    
    # Simple summary
    sentences = text.split('.')[:5]
    langchain_summary = '. '.join(sentences) + '.'
    langchain_insights = f"Key insights: {langchain_summary[:150]}..."
    
    # Generate comprehensive insights summary
    insights_summary = f"""
    DOCUMENT INSIGHTS SUMMARY:
    
    {spacy_summary}
    
    {topic_summary}
    
    {transformer_summary}
    
    {langchain_insights}
    """
    
    # Create detailed insights dictionary
    detailed_insights = {
        'file_name': file_name,
        'spacy': spacy_insights,
        'topics': topics,
        'transformer': transformer_results,
        'langchain': langchain_summary,
        'saved_models': {
            'topic_model': topic_model_file,
            'keras_model': keras_model_file
        }
    }
    
    # Save detailed insights
    with open('document_insights.pkl', 'wb') as f:
        pickle.dump(detailed_insights, f)
    
    return insights_summary.strip(), detailed_insights

# Process PDF document - optimized
def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    start_time = time.time()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Try PyPDF2 first - fastest method
    status_text.text("Extracting text from PDF...")
    pypdf_text = extract_text_with_pyPDF(file_bytes)
    progress_bar.progress(30)
    
    # Step 2: Only use OCR if PyPDF2 doesn't get enough text
    combined_text = pypdf_text
    images = []
    
    if len(pypdf_text.strip().split()) < 100:  # If PyPDF2 gets very little text
        status_text.text("Converting PDF to images...")
        images = convert_pdf_to_images(file_bytes, max_pages=3)  # Limit to 3 pages
        progress_bar.progress(50)
        
        status_text.text("Extracting text from images...")
        pytesseract_text = extract_text_with_pytesseract(images)
        combined_text = pypdf_text + "\n" + pytesseract_text
        progress_bar.progress(70)
    else:
        # Skip OCR for speed if we already have text
        status_text.text("Sufficient text extracted from PDF. Skipping image processing...")
        progress_bar.progress(70)
    
    # Step 3: Generate insights
    status_text.text("Generating insights...")
    insights, detailed_insights = generate_comprehensive_insights(combined_text, uploaded_file.name)
    progress_bar.progress(100)
    
    end_time = time.time()
    processing_time = end_time - start_time
    status_text.text(f"Processing complete in {processing_time:.2f} seconds!")
    
    return {
        'file_name': uploaded_file.name,
        'text': {
            'pypdf': pypdf_text,
            'pytesseract': pytesseract_text if 'pytesseract_text' in locals() else "Not performed",
        },
        'images': images,
        'insights': insights,
        'detailed_insights': detailed_insights,
        'processing_time': processing_time
    }

# Process image - optimized
def process_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    start_time = time.time()
    image = Image.open(BytesIO(file_bytes))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create image dict in same format as PDF processing
    image_dict = {0: file_bytes}
    images = [image_dict]
    progress_bar.progress(20)
    
    # Extract text using PyTesseract only
    status_text.text("Extracting text from image...")
    import pytesseract
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        pass
    
    pytesseract_text = pytesseract.image_to_string(image)
    progress_bar.progress(70)
    
    # Generate insights
    status_text.text("Generating insights...")
    insights, detailed_insights = generate_comprehensive_insights(pytesseract_text, uploaded_file.name)
    progress_bar.progress(100)
    
    end_time = time.time()
    processing_time = end_time - start_time
    status_text.text(f"Processing complete in {processing_time:.2f} seconds!")
    
    return {
        'file_name': uploaded_file.name,
        'text': {
            'pytesseract': pytesseract_text,
        },
        'images': images,
        'insights': insights,
        'detailed_insights': detailed_insights,
        'processing_time': processing_time
    }

# Function to create download link for saved models
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Display PDF images
def display_pdf_images(images):
    if not images:
        st.info("No images processed (text extraction was sufficient)")
        return
        
    cols = st.columns(min(3, len(images)))
    for i, image_dict in enumerate(images):
        image_bytes = list(image_dict.values())[0]
        image = Image.open(BytesIO(image_bytes))
        cols[i % 3].image(image, caption=f"Page {i+1}", use_column_width=True)

# Main function that creates the Streamlit interface
def main():
    st.title("📄 Fast Document Insights Extractor")
    st.markdown("""
    Upload a PDF document or image to extract text and generate comprehensive insights in under a minute!
    
    This optimized version:
    - Prioritizes speed over exhaustive analysis
    - Uses spaCy for fast entity recognition and key phrases
    - Performs lightweight topic modeling
    - Uses rule-based sentiment analysis
    - Only processes OCR if text extraction fails
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Check if it's a PDF or image
        file_type = uploaded_file.type
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Processing: {uploaded_file.name}")
            st.markdown(f"**File type:** {file_type}")
        
        # Process file
        if 'pdf' in file_type:
            results = process_pdf(uploaded_file)
        else:  # Image
            results = process_image(uploaded_file)
        
        st.success(f"Processing completed in {results['processing_time']:.2f} seconds!")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Insights", "📝 Extracted Text", "🖼️ Images", "💾 Models"])
        
        with tab1:
            st.markdown("### Document Insights")
            st.markdown(results['insights'])
            
            # Display entity visualization
            st.subheader("Named Entities")
            entities = results['detailed_insights']['spacy']['entities']
            if entities:
                try:
                    import pandas as pd
                    entity_df = pd.DataFrame(list(entities.items()), columns=['Entity', 'Type'])
                    st.dataframe(entity_df)
                except ImportError:
                    st.write(entities)
            else:
                st.info("No named entities detected")
            
            # Display topics
            st.subheader("Topic Analysis")
            topics = results['detailed_insights']['topics']
            if topics:
                for i, topic in enumerate(topics):
                    st.markdown(f"**Topic {i+1}:** {', '.join(topic[:10])}")
            else:
                st.info("No significant topics detected")
            
            # Display sentiment
            st.subheader("Sentiment Analysis")
            sentiment = results['detailed_insights']['transformer']['sentiment']
            st.markdown(f"**Overall sentiment:** {sentiment}")
            
            # Display summary
            st.subheader("Document Summary")
            summary = results['detailed_insights']['transformer']['summary']
            st.markdown(summary)
        
        with tab2:
            st.markdown("### Extracted Text")
            for method, text in results['text'].items():
                with st.expander(f"Text extracted using {method.upper()}"):
                    st.text_area("", text, height=200)
        
        with tab3:
            st.markdown("### Document Images")
            if 'pdf' in file_type:
                display_pdf_images(results['images'])
            else:
                image_bytes = list(results['images'][0].values())[0]
                st.image(Image.open(BytesIO(image_bytes)), caption="Uploaded Image")
        
        with tab4:
            st.markdown("### Download Saved Models")
            
            # Topic model
            if os.path.exists('topic_model.pkl'):
                st.markdown(
                    get_binary_file_downloader_html('topic_model.pkl', 'Download Topic Model (.pkl)'),
                    unsafe_allow_html=True
                )
            
            # Insights
            if os.path.exists('document_insights.pkl'):
                st.markdown(
                    get_binary_file_downloader_html('document_insights.pkl', 'Download Document Insights (.pkl)'),
                    unsafe_allow_html=True
                )

# Only run the app if this is the main script
if __name__ == "__main__":
    main()