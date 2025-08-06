import clip
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os

class CLIPMedicalClassifier:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize CLIP-based medical image classifier
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run model on (auto-detect if None)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Define optimized text prompts for medical vs non-medical classification
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
        
        # Tokenize all text prompts
        self._setup_text_features()
        
    def _setup_text_features(self):
        """Setup text features for all prompts"""
        print("Setting up text features...")
        
        # Create text inputs
        all_descriptions = self.medical_descriptions + self.non_medical_descriptions
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of {desc}") for desc in all_descriptions
        ]).to(self.device)
        
        # Encode text features
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features = F.normalize(self.text_features, dim=-1)
        
        print(f"Encoded {len(all_descriptions)} text prompts")
        
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
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
            prediction = 0 if medical_prob > non_medical_prob else 1  # 0=medical, 1=non-medical
            confidence = max(medical_prob, non_medical_prob)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'medical_prob': medical_prob,
                'non_medical_prob': non_medical_prob,
                'class_name': 'medical' if prediction == 0 else 'non-medical'
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict classes for a batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in tqdm(image_paths, desc="Predicting"):
            result = self.predict_single_image(image_path)
            if result:
                results.append(result)
        return results
    
    def get_image_paths(self, folder: str) -> List[str]:
        """Get all image paths from a folder"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(supported_formats)]
    
    def evaluate_split(self, image_paths: List[str], true_labels: List[int], 
                      split_name: str = "test") -> Dict:
        """
        Evaluate model performance on a dataset split
        
        Args:
            image_paths: List of image paths
            true_labels: List of true labels (0=medical, 1=non-medical)
            split_name: Name of the split for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {split_name} split...")
        
        # Get predictions
        predictions = self.predict_batch(image_paths)
        
        if not predictions:
            print(f"No valid predictions for {split_name} split")
            return {}
        
        # Extract predictions and confidences
        pred_labels = [p['prediction'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Classification report
        report = classification_report(
            true_labels, pred_labels, 
            target_names=['medical', 'non-medical'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        results = {
            'split': split_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'confidences': confidences
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, 
                                 target_names=['medical', 'non-medical']))
        
        return results
    
    def evaluate_full_dataset(self, data_dir: str) -> Dict:
        """
        Evaluate model on the full dataset with predefined train, val, test splits.

        Args:
            data_dir: Path to root dataset directory containing 'train', 'val', 'test' folders

        Returns:
            Dictionary with evaluation results for each split
        """
        results = {}

        for split in ['train','val','test']:
            split_path = os.path.join(data_dir, split)
            medical_dir = os.path.join(split_path, 'Medical')
            non_medical_dir = os.path.join(split_path, 'Non-Medical')

            if not os.path.exists(medical_dir) or not os.path.exists(non_medical_dir):
                print(f"Skipping {split.upper()} - Missing directories")
                continue

            medical_paths = self.get_image_paths(medical_dir)
            non_medical_paths = self.get_image_paths(non_medical_dir)

            all_paths = medical_paths + non_medical_paths
            true_labels = [0] * len(medical_paths) + [1] * len(non_medical_paths)

            if all_paths:
                print(f"\n{split.upper()} Split:")
                print(f"  Medical images: {len(medical_paths)}")
                print(f"  Non-Medical images: {len(non_medical_paths)}")
                print(f"  Total images: {len(all_paths)}")

                results[split] = self.evaluate_split(all_paths, true_labels, split)
            else:
                print(f"No images found in {split} split")

        return results
    
    def save_results(self, results: Dict, output_file: str = "clip_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj  # default fallback
        
        serializable_results = {
            split: convert_to_serializable(split_result)
            for split, split_result in results.items()
        }

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {output_file}")
    
    def plot_confusion_matrix(self, results: Dict, output_dir: str = "plots"):
        """Plot confusion matrices for all splits"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split, split_results in results.items():
            if 'confusion_matrix' in split_results:
                cm = np.array(split_results['confusion_matrix'])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['medical', 'non-medical'],
                           yticklabels=['medical', 'non-medical'])
                plt.title(f'Confusion Matrix - {split.upper()} Split')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/confusion_matrix_{split}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Confusion matrix saved: {output_dir}/confusion_matrix_{split}.png")

    def save_model(self, output_dir: str = "saved_models"):
        """Save the CLIP model and related components"""
        os.makedirs(output_dir, exist_ok=True)    
        model_path = os.path.join(output_dir, "clip_model.pth")

        # Save model state
        torch.save({
        'model_state_dict': self.model.state_dict(),
        'text_features': self.text_features,
        'medical_descriptions': self.medical_descriptions,
        'non_medical_descriptions': self.non_medical_descriptions
        }, model_path)
    
        print(f"Model saved to {model_path}")



def main():
    parser = argparse.ArgumentParser(description='CLIP-based Medical Image Classifier')
    parser.add_argument('--data_dir', default='ProcessedDataSet', 
                       help='Path to preprocessed dataset directory')
    parser.add_argument('--model_name', default='ViT-B/32', 
                       help='CLIP model variant to use')
    parser.add_argument('--output_file', default='clip_evaluation_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--plots_dir', default='plots',
                       help='Directory for saving plots')
    parser.add_argument('--model_dir', default='saved_models',
                       help='Directory for saving the model')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = CLIPMedicalClassifier(model_name=args.model_name)
    
    # Evaluate dataset
    print("Starting evaluation...")
    results = classifier.evaluate_full_dataset(args.data_dir)
    
    # Save results
    classifier.save_results(results, args.output_file)
    
    # Create plots
    classifier.plot_confusion_matrix(results, args.plots_dir)
    
    # Save model
    classifier.save_model(args.model_dir)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {args.output_file}")
    print(f"Plots saved to: {args.plots_dir}/")
    print(f"Model saved to: {args.model_dir}/")

if __name__ == "__main__":
    main() 