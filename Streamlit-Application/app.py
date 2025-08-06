import streamlit as st
import torch
import clip
import torch.nn.functional as F
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import fitz  # PyMuPDF
import tempfile
import os
from pathlib import Path
import urllib.parse
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple
import time

class MediCLIPClassifier:
    def __init__(self, model_path: str = "Scripts/saved_models/clip_model.pth"):
        """Initialize the MediCLIP classifier"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load custom weights if available
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                st.success(f"✅ Loaded custom model from {model_path}")
            except Exception as e:
                st.warning(f"⚠️ Could not load custom model: {e}. Using default CLIP model.")
        else:
            st.info("ℹ️ Using default CLIP model (no custom weights found)")
        
        # Define medical and non-medical descriptions
        self.medical_descriptions = [
            "a medical image", "an X-ray scan", "an MRI scan", 
            "a CT scan", "an ultrasound image", "a histopathology slide",
            "a medical diagnostic image", "a clinical photograph",
            "a medical examination image", "a healthcare image",
            "a radiology image", "a medical test result",
            "a patient scan", "a medical procedure image"
        ]
        
        self.non_medical_descriptions = [
            "a natural image", "a landscape", "an animal", 
            "a building", "a cityscape", "a household object",
            "a nature photograph", "a street scene",
            "a portrait", "a food image", "a vehicle",
            "a sports image", "a fashion image", "an art piece"
        ]
        
        # Setup text features
        self._setup_text_features()
    
    def _setup_text_features(self):
        """Setup text features for classification"""
        all_descriptions = self.medical_descriptions + self.non_medical_descriptions
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of {desc}") for desc in all_descriptions
        ]).to(self.device)
        
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features = F.normalize(self.text_features, dim=-1)
    
    def classify_image(self, image: Image.Image) -> Dict:
        """Classify a single image"""
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarities
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                probs = similarity.cpu().numpy()[0]
            
            # Aggregate probabilities
            medical_prob = np.sum(probs[:len(self.medical_descriptions)])
            non_medical_prob = np.sum(probs[len(self.medical_descriptions):])
            
            # Get prediction
            prediction = 0 if medical_prob > non_medical_prob else 1
            confidence = max(medical_prob, non_medical_prob)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'medical_prob': float(medical_prob),
                'non_medical_prob': float(non_medical_prob),
                'class_name': 'medical' if prediction == 0 else 'non-medical'
            }
            
        except Exception as e:
            st.error(f"Error classifying image: {e}")
            return None

class ImageExtractor:
    def __init__(self):
        """Initialize image extractor with enhanced headers and settings"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _is_valid_image(self, content_type: str, url: str) -> bool:
        """Check if the content type is a valid image"""
        valid_types = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp']
        if any(t in content_type.lower() for t in valid_types):
            return True
        
        # Additional check for URLs with image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp']
        if any(url.lower().endswith(ext) for ext in image_extensions):
            return True
            
        return False
    
    def _get_robots_txt(self, base_url: str) -> List[str]:
        """Check robots.txt for scraping permissions"""
        try:
            parsed_url = urllib.parse.urlparse(base_url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                return response.text.splitlines()
        except Exception:
            pass
        return []
    
    def _is_scraping_allowed(self, url: str) -> bool:
        """Check if scraping is allowed for the given URL"""
        robots_rules = self._get_robots_txt(url)
        if not robots_rules:
            return True
            
        user_agent = self.headers['User-Agent'].split('/')[0]
        disallowed_paths = []
        
        for line in robots_rules:
            if line.lower().startswith('user-agent:') and (line[11:].strip() == '*' or user_agent in line[11:]):
                current_user_agent = line[11:].strip()
            elif line.lower().startswith('disallow:') and 'current_user_agent' in locals():
                disallowed_paths.append(line[9:].strip())
        
        parsed_url = urllib.parse.urlparse(url)
        for path in disallowed_paths:
            if parsed_url.path.startswith(path):
                return False
        return True
    
    def extract_images_from_url(self, url: str) -> List[Tuple[Image.Image, str]]:
        """Extract images from a website URL with enhanced scraping capabilities"""
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Check if scraping is allowed
            if not self._is_scraping_allowed(url):
                st.warning(f"⚠️ Scraping is restricted for {url} according to robots.txt")
                return []
            
            # Fetch webpage with enhanced error handling
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                
                # Check if content is HTML
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    st.error("URL does not point to an HTML page")
                    return []
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    st.warning(f"⚠️ Access denied (403) for {url}. Some websites block automated requests.")
                    st.info("ℹ️ Try these solutions:")
                    st.markdown("- Use a different website that allows scraping")
                    st.markdown("- Check if the website has a public API you can use instead")
                    st.markdown("- Contact the website owner for permission")
                    return []
                else:
                    st.error(f"Failed to fetch webpage: {e}")
                    return []
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch webpage: {e}")
                return []
            
            # Parse HTML with fallback parsers
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                st.error(f"Failed to parse HTML: {e}")
                return []
            
            images = []
            
            # Find all img tags and picture sources
            img_tags = soup.find_all('img')
            picture_tags = soup.find_all('picture')
            
            # Extract sources from picture tags
            for pic in picture_tags:
                sources = pic.find_all('source')
                for src in sources:
                    img_src = src.get('srcset') or src.get('data-srcset')
                    if img_src:
                        img_tags.append(src)  # Treat as img tag for processing
            
            for i, img_tag in enumerate(img_tags):
                # Try multiple attributes for image source
                img_src = (img_tag.get('src') or 
                          img_tag.get('data-src') or 
                          img_tag.get('data-lazy-src') or
                          img_tag.get('data-original') or
                          (img_tag.get('srcset', '').split(',')[0].split()[0] if img_tag.get('srcset') else None))
                
                if not img_src:
                    continue
                    
                # Handle relative URLs and different URL formats
                try:
                    if img_src.startswith('//'):
                        img_src = 'https:' + img_src
                    elif img_src.startswith('/'):
                        parsed_url = urllib.parse.urlparse(url)
                        img_src = f"{parsed_url.scheme}://{parsed_url.netloc}{img_src}"
                    elif not img_src.startswith(('http://', 'https://')):
                        img_src = urllib.parse.urljoin(url, img_src)
                    
                    # Skip SVG and non-image URLs
                    if img_src.lower().endswith('.svg'):
                        continue
                    if 'placeholder' in img_src.lower():
                        continue
                    
                    # Download image with enhanced error handling
                    try:
                        img_response = self.session.get(img_src, timeout=15, stream=True)
                        img_response.raise_for_status()
                        
                        # Verify content is actually an image
                        content_type = img_response.headers.get('content-type', '')
                        if not self._is_valid_image(content_type, img_src):
                            continue
                            
                        # Convert to PIL Image
                        try:
                            img_data = BytesIO(img_response.content)
                            img = Image.open(img_data).convert('RGB')
                            images.append((img, f"Image {i+1} from {url}"))
                        except Exception as img_error:
                            continue
                            
                    except requests.exceptions.RequestException:
                        continue
                    
                except Exception:
                    continue
            
            if not images:
                st.warning("⚠️ No images found. Possible reasons:")
                st.markdown("- The website uses JavaScript to load images (try a different site)")
                st.markdown("- Images are behind a login wall")
                st.markdown("- The website blocks image scraping")
                st.markdown("- All images were filtered out (e.g., SVGs, placeholders)")
            
            return images
            
        except Exception as e:
            st.error(f"Unexpected error extracting images from URL: {e}")
            return []

    def extract_images_from_pdf(self, pdf_file) -> List[Tuple[Image.Image, str]]:
        """Extract images from uploaded PDF file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Open PDF
            doc = fitz.open(tmp_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get images from page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Convert to PIL Image
                        img_pil = Image.open(BytesIO(img_data)).convert('RGB')
                        images.append((img_pil, f"Page {page_num+1}, Image {img_index+1}"))
                        
                        pix = None
                        
                    except Exception as e:
                        st.warning(f"Could not extract image {img_index+1} from page {page_num+1}: {e}")
            
            doc.close()
            os.unlink(tmp_path)  # Clean up temp file
            
            return images
            
        except Exception as e:
            st.error(f"Error extracting images from PDF: {e}")
            return []

def main():
    st.set_page_config(
        page_title="MediCLIP - Medical Image Classifier",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 MediCLIP - Medical Image Classifier")
    st.markdown("Classify images as medical or non-medical using CLIP-based AI")
    
    # Initialize classifier
    with st.spinner("Loading CLIP model..."):
        classifier = MediCLIPClassifier()
    
    # Sidebar for input options
    st.sidebar.header("📥 Input Options")
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Website URL", "PDF Upload", "Single Image Upload"]
    )
    
    # Main content area
    if input_method == "Website URL":
        st.header("🌐 Extract Images from Website")
        
        # URL input
        url = st.text_input(
            "Enter website URL:",
            placeholder="https://example.com or www.example.com",
            help="Enter a website URL to extract and classify images"
        )
        
        # Example URLs for testing
        st.markdown("**Example URLs for testing:**")
        st.markdown("- Medical: https://www.radiologyinfo.org/en/info/")
        st.markdown("- General: https://unsplash.com/")
        
        if st.button("🔍 Extract and Classify Images", type="primary"):
            if url:
                with st.spinner("Extracting images from website..."):
                    extractor = ImageExtractor()
                    images = extractor.extract_images_from_url(url)
                    
                    if images:
                        st.success(f"✅ Found {len(images)} images")
                        classify_images(images, classifier)
                    else:
                        st.warning("⚠️ No images found on the website")
            else:
                st.error("Please enter a URL")
    
    elif input_method == "PDF Upload":
        st.header("📄 Extract Images from PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file:",
            type=['pdf'],
            help="Upload a PDF file to extract and classify images"
        )
        
        if uploaded_file is not None:
            if st.button("🔍 Extract and Classify Images", type="primary"):
                with st.spinner("Extracting images from PDF..."):
                    extractor = ImageExtractor()
                    images = extractor.extract_images_from_pdf(uploaded_file)
                    
                    if images:
                        st.success(f"✅ Found {len(images)} images")
                        classify_images(images, classifier)
                    else:
                        st.warning("⚠️ No images found in PDF")
    
    else:  # Single Image Upload
        st.header("🖼️ Classify Single Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image file:",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload a single image to classify"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_image).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Classify Image", type="primary"):
                    result = classifier.classify_image(image)
                    if result:
                        display_classification_result(result, "Single Image")
            except Exception as e:
                st.error(f"Could not open image: {e}")

def classify_images(images: List[Tuple[Image.Image, str]], classifier: MediCLIPClassifier):
    """Classify multiple images and display results"""
    st.header("📊 Classification Results")
    
    # Create columns for results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Medical Images")
        medical_images = []
    
    with col2:
        st.subheader("Non-Medical Images")
        non_medical_images = []
    
    # Process each image
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (image, description) in enumerate(images):
        status_text.text(f"Processing image {i+1}/{len(images)}: {description}")
        
        result = classifier.classify_image(image)
        if result:
            # Resize image for display
            display_image = image.copy()
            display_image.thumbnail((300, 300))
            
            if result['class_name'] == 'medical':
                medical_images.append((display_image, description, result))
            else:
                non_medical_images.append((display_image, description, result))
        
        progress_bar.progress((i + 1) / len(images))
    
    status_text.text("✅ Classification complete!")
    
    # Display results
    with col1:
        if medical_images:
            for img, desc, result in medical_images:
                st.image(img, caption=f"{desc} (Confidence: {result['confidence']:.2f})")
        else:
            st.info("No medical images found")
    
    with col2:
        if non_medical_images:
            for img, desc, result in non_medical_images:
                st.image(img, caption=f"{desc} (Confidence: {result['confidence']:.2f})")
        else:
            st.info("No non-medical images found")
    
    # Summary statistics
    st.header("📈 Summary")
    total_images = len(images)
    medical_count = len(medical_images)
    non_medical_count = len(non_medical_images)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Medical Images", medical_count)
    with col3:
        st.metric("Non-Medical Images", non_medical_count)

def display_classification_result(result: Dict, image_name: str):
    """Display classification result for a single image"""
    st.header("🎯 Classification Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Classification", result['class_name'].title())
        st.metric("Confidence", f"{result['confidence']:.2%}")
    
    with col2:
        st.metric("Medical Probability", f"{result['medical_prob']:.2%}")
        st.metric("Non-Medical Probability", f"{result['non_medical_prob']:.2%}")
    
    # Progress bars
    st.subheader("Probability Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(result['medical_prob'])
        st.caption("Medical")
    
    with col2:
        st.progress(result['non_medical_prob'])
        st.caption("Non-Medical")

if __name__ == "__main__":
    main()