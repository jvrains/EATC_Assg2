# copy_model_files.py - Run this script in your project directory

import os
import shutil

def copy_model_files():
    """Copy model files from Jupyter location to Streamlit location"""
    
    # Create directories
    os.makedirs('models/finetuned', exist_ok=True)
    os.makedirs('models/pretrained', exist_ok=True)
    os.makedirs('models/encoders', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Define file mappings (source -> destination)
    file_mappings = {
        # Main models
        'assignment2/models/finetuned/enhanced_ddos_model.pkl': 'models/finetuned/enhanced_ddos_model.pkl',
        'assignment2/models/pretrained/baseline_model.pkl': 'models/pretrained/baseline_model.pkl',
        
        # Metadata
        'assignment2/models/finetuned/model_metadata.json': 'models/finetuned/model_metadata.json',
        'assignment2/models/finetuned/feature_importance.csv': 'models/finetuned/feature_importance.csv',
        
        # Sample data
        'assignment2/data/sample_network_traffic.csv': 'data/sample_network_traffic.csv',
        'assignment2/data/sample_scenarios.json': 'data/sample_scenarios.json',
    }
    
    # Copy individual files
    for source, dest in file_mappings.items():
        if os.path.exists(source):
            shutil.copy2(source, dest)
            print(f"✅ Copied: {source} -> {dest}")
        else:
            print(f"❌ Not found: {source}")
    
    # Copy all encoder files
    encoder_source_dir = '../models/encoders'
    encoder_dest_dir = 'models/encoders'
    
    if os.path.exists(encoder_source_dir):
        for filename in os.listdir(encoder_source_dir):
            if filename.endswith('.pkl'):
                source = os.path.join(encoder_source_dir, filename)
                dest = os.path.join(encoder_dest_dir, filename)
                shutil.copy2(source, dest)
                print(f"✅ Copied encoder: {filename}")
    
    print("\n🎉 File copying complete!")
    print("Now restart your Streamlit app.")

if __name__ == "__main__":
    copy_model_files()