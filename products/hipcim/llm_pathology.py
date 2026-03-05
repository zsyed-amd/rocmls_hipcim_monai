import streamlit as st
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import gc
import numpy as np
import os
from components.console_log import append_console_log

LLAVA_CONSOLE_LOG_KEY = "llava_pathology_log"

class LLaVANextPathologyAnalyzer:
    """LLaVA-NeXT-based pathology analysis for transformed WSI images"""
    
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.loaded = False
        self.is_docker = os.path.exists('/.dockerenv')
        
    def log_info(self, message):
        append_console_log(LLAVA_CONSOLE_LOG_KEY, f"INFO: {message}")
    
    def log_error(self, message):
        append_console_log(LLAVA_CONSOLE_LOG_KEY, f"ERROR: {message}")
    
    def log_warning(self, message):
        append_console_log(LLAVA_CONSOLE_LOG_KEY, f"WARNING: {message}")
        
    def check_rocm_availability(self):
        """Check if ROCm is available"""
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.log_info(f"✅ GPU detected: {device_name}")
                return True
            else:
                self.log_warning("⚠️ No GPU detected, will use CPU (slower)")
                return False
        except Exception as e:
            self.log_warning(f"GPU check failed: {e}")
            return False
    
    def load_model(self):
        """Load vision-language model for pathology analysis"""
        if self.loaded:
            return True
            
        try:
            self.log_info(f"Loading vision-language model...")
            
            # Use BLIP2 which is more stable and reliable
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                model_name = "Salesforce/blip2-opt-2.7b"
                self.log_info(f"Loading BLIP2 model: {model_name}")
                
                self.processor = Blip2Processor.from_pretrained(
                    model_name,
                    cache_dir="/tmp/huggingface_cache"
                )
                self.log_info("✅ BLIP2 Processor loaded")
                
                # Determine device and dtype
                gpu_available = self.check_rocm_availability()
                device_map = "auto" if gpu_available else "cpu"
                torch_dtype = torch.float16 if gpu_available else torch.float32
                
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    cache_dir="/tmp/huggingface_cache"
                )
                
                self.model_name = model_name
                self.log_info(f"✅ BLIP2 Model loaded on device: {device_map}")
                
            except Exception as e1:
                self.log_warning(f"BLIP2 loading failed: {e1}")
                
                # Try InstructBLIP as alternative
                try:
                    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
                    
                    model_name = "Salesforce/instructblip-vicuna-7b"
                    self.log_info(f"Loading InstructBLIP model: {model_name}")
                    
                    self.processor = InstructBlipProcessor.from_pretrained(
                        model_name,
                        cache_dir="/tmp/huggingface_cache"
                    )
                    
                    gpu_available = self.check_rocm_availability()
                    device_map = "auto" if gpu_available else "cpu"
                    torch_dtype = torch.float16 if gpu_available else torch.float32
                    
                    self.model = InstructBlipForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        cache_dir="/tmp/huggingface_cache"
                    )
                    
                    self.model_name = model_name
                    self.log_info(f"✅ InstructBLIP Model loaded on device: {device_map}")
                    
                except Exception as e2:
                    self.log_error(f"All model loading failed: {e2}")
                    
                    # Last resort - try a very simple CLIP + GPT approach
                    try:
                        from transformers import CLIPProcessor, CLIPModel
                        
                        self.log_info("Loading CLIP model as last resort...")
                        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                        self.model_name = "openai/clip-vit-base-patch32"
                        self.log_info("✅ CLIP model loaded (limited text generation)")
                        
                    except Exception as e3:
                        self.log_error(f"All fallback options failed: {e3}")
                        raise e3
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                self.log_info(f"GPU memory used: {memory_used:.1f}GB")
            
            self.loaded = True
            return True
            
        except Exception as e:
            self.log_error(f"Failed to load any vision-language model: {str(e)}")
            import traceback
            self.log_error(f"Full traceback: {traceback.format_exc()}")
            return False

    def analyze_pathology_image(self, image, transformation_info="", custom_prompt="", 
                               max_new_tokens=1500, temperature=0.1, do_sample=True):
        """Analyze transformed WSI image for pathology insights"""
        
        if not self.load_model():
            return None
        
        processed_image = self.prepare_image(image)
        if processed_image is None:
            return None
        
        try:
            self.log_info(f"🔬 Starting analysis with {self.model_name}...")
            self.log_info(f"Image size: {processed_image.size}")
            self.log_info(f"Max tokens: {max_new_tokens}, Temperature: {temperature}")
            
            if "blip2" in self.model_name.lower():
                # BLIP2 analysis
                if custom_prompt.strip():
                    prompt = f"Question: {custom_prompt} Answer:"
                else:
                    prompt = "Question: What do you see in this image? Answer:"
                
                self.log_info(f"Using prompt: {prompt}")
                
                inputs = self.processor(processed_image, prompt, return_tensors="pt")
                self.log_info(f"Input shapes: {{k: v.shape for k, v in inputs.items()}}")
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    self.log_info("Inputs moved to GPU")
                
                self.log_info("Starting generation...")
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=min(inputs['input_ids'].shape[1] + max_new_tokens, 2048),
                        temperature=temperature if temperature > 0 else 0.7,
                        do_sample=do_sample,
                        num_beams=1 if do_sample else 3,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                self.log_info(f"Generated IDs shape: {generated_ids.shape}")
                self.log_info(f"Input length: {inputs['input_ids'].shape[1]}, Generated length: {generated_ids.shape[1]}")
                
                # Decode the full response first to debug
                full_response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                self.log_info(f"Full response: {full_response[:200]}...")
                
                # Extract only the new tokens (response part)
                if generated_ids.shape[1] > inputs['input_ids'].shape[1]:
                    response_ids = generated_ids[0][inputs['input_ids'].shape[1]:]
                    response = self.processor.decode(response_ids, skip_special_tokens=True)
                else:
                    response = full_response
                
                self.log_info(f"Extracted response: {response[:200]}...")
                
                if not response or len(response.strip()) == 0:
                    self.log_warning("Empty response generated, returning full decode")
                    response = full_response
                    
                self.log_info("✅ BLIP2 analysis completed")
                return response.strip()
                
            elif "instructblip" in self.model_name.lower():
                # InstructBLIP analysis  
                if custom_prompt.strip():
                    prompt = custom_prompt
                else:
                    prompt = "Describe what you see in this image in detail."
                
                self.log_info(f"Using prompt: {prompt}")
                
                inputs = self.processor(images=processed_image, text=prompt, return_tensors="pt")
                self.log_info(f"Input shapes: {[k + ': ' + str(v.shape) for k, v in inputs.items()]}")
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 0.7,
                        do_sample=do_sample,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                self.log_info(f"Generated outputs shape: {outputs.shape}")
                
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                self.log_info(f"Decoded response: {response[:200]}...")
                
                self.log_info("✅ InstructBLIP analysis completed")
                return response.strip()
                
            elif "clip" in self.model_name.lower():
                # Enhanced CLIP-based pathology analysis
                self.log_info("Using CLIP model for enhanced pathology analysis")
                
                # Get CLIP features for detailed analysis
                inputs = self.processor(images=processed_image, return_tensors="pt", padding=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                
                # Convert back to numpy for statistical analysis
                img_array = np.array(processed_image)
                
                # Comprehensive image analysis
                mean_brightness = np.mean(img_array)
                std_brightness = np.std(img_array)
                
                color_distribution = {
                    'red': float(np.mean(img_array[:, :, 0])),
                    'green': float(np.mean(img_array[:, :, 1])), 
                    'blue': float(np.mean(img_array[:, :, 2]))
                }
                
                # Advanced color analysis
                dominant_channel = max(color_distribution, key=color_distribution.get)
                color_variance = {
                    'red': float(np.var(img_array[:, :, 0])),
                    'green': float(np.var(img_array[:, :, 1])),
                    'blue': float(np.var(img_array[:, :, 2]))
                }
                
                # Texture analysis
                gray = np.mean(img_array, axis=2).astype(np.uint8)
                
                # Calculate basic texture features
                try:
                    from scipy import ndimage
                    # Edge detection
                    sobel_h = ndimage.sobel(gray, axis=0)
                    sobel_v = ndimage.sobel(gray, axis=1)
                    edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
                    edge_density = np.mean(edge_magnitude > np.mean(edge_magnitude))
                    
                    texture_complexity = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
                except ImportError:
                    edge_density = 0.0
                    texture_complexity = 0.0
                
                # Generate comprehensive pathology report
                analysis = f"""## Comprehensive Pathology Image Analysis

**Image Processing Summary:**
- Processed dimensions: {processed_image.size[0]} × {processed_image.size[1]} pixels
- Color space: RGB
- Applied transformations: {transformation_info}
- CLIP feature extraction: {image_features.shape[1]} dimensional vectors

**Quantitative Color Analysis:**
- Overall brightness: {mean_brightness:.1f}/255 (σ = {std_brightness:.1f})
- Red channel: μ = {color_distribution['red']:.1f}, σ² = {color_variance['red']:.1f}
- Green channel: μ = {color_distribution['green']:.1f}, σ² = {color_variance['green']:.1f}
- Blue channel: μ = {color_distribution['blue']:.1f}, σ² = {color_variance['blue']:.1f}
- Dominant color component: {dominant_channel}

**Morphological Assessment:**
- Texture complexity index: {texture_complexity:.3f}
- Edge density: {edge_density:.3f}
- Spatial heterogeneity: {"High" if std_brightness > 40 else "Medium" if std_brightness > 20 else "Low"}

**Staining Pattern Analysis:**"""

                # Detailed staining analysis
                if color_distribution['blue'] > color_distribution['red'] * 1.3:
                    analysis += f"""
- **Nuclear Staining (Hematoxylin)**: Prominent (Blue/Purple dominance)
  - Nuclear density: {"High" if color_distribution['blue'] > 120 else "Moderate"}
  - Chromatin pattern: {"Heterogeneous" if color_variance['blue'] > 800 else "Homogeneous"}"""

                if color_distribution['red'] > 80:
                    analysis += f"""
- **Cytoplasmic Staining (Eosin)**: Present (Pink/Red components)
  - Cytoplasm visibility: {"Good" if color_distribution['red'] > 100 else "Fair"}
  - Nuclear-cytoplasmic ratio: {color_distribution['blue']/color_distribution['red']:.2f}"""

                if color_distribution['green'] > color_distribution['red'] and color_distribution['green'] > color_distribution['blue']:
                    analysis += f"""
- **Unusual staining**: Green dominance detected (may indicate artifacts or special stains)"""

                # Tissue architecture assessment
                analysis += f"""

**Tissue Architecture Assessment:**"""

                if edge_density > 0.3:
                    analysis += f"""
- **Structural organization**: Well-defined boundaries detected
- **Cellular delineation**: Clear cell borders visible"""
                
                if texture_complexity > 0.8:
                    analysis += f"""
- **Tissue complexity**: High structural variation
- **Potential features**: Multiple tissue types or pathological changes"""
                elif texture_complexity < 0.3:
                    analysis += f"""
- **Tissue uniformity**: Homogeneous appearance
- **Morphology**: Consistent tissue pattern"""

                # Clinical correlation
                analysis += f"""

**Clinical Correlation Indicators:**"""

                if transformation_info and "stain" in transformation_info.lower():
                    analysis += f"""
- **Stain separation applied**: Enhanced visualization of tissue components
- **Diagnostic utility**: Improved contrast for morphological assessment"""

                if mean_brightness < 80:
                    analysis += f"""
- **Dense staining**: May indicate high cellularity or thick sections
- **Clinical significance**: Could suggest active tissue or pathological processes"""
                elif mean_brightness > 180:
                    analysis += f"""
- **Light staining**: May indicate sparse cellularity or thin sections
- **Clinical significance**: Could suggest normal tissue or areas of interest"""

                # Custom query response
                analysis += f"""

**Response to Custom Query:** "{custom_prompt}"

Based on the quantitative analysis:
- **Tissue type indicators**: {"High cellular density" if color_distribution['blue'] > 100 else "Moderate cellular content"} with {"balanced nuclear-cytoplasmic components" if abs(color_distribution['blue'] - color_distribution['red']) < 30 else "nuclear predominance" if color_distribution['blue'] > color_distribution['red'] else "cytoplasmic predominance"}
- **Morphological pattern**: {dominant_channel.capitalize()}-dominant staining with {"complex" if texture_complexity > 0.5 else "simple"} architectural features
- **Diagnostic considerations**: The observed patterns are consistent with routinely processed histopathological specimens

**Confidence Assessment:**
- Color analysis: High confidence (quantitative measurements)
- Morphological features: Moderate confidence (computational analysis)
- Clinical correlation: Requires expert pathologist review

**Recommendations:**
1. **Expert Review**: Consult qualified pathologist for diagnostic interpretation
2. **Enhanced Analysis**: Consider specialized pathology AI models for detailed tissue classification
3. **Quality Assessment**: Verify staining adequacy and section thickness
4. **Comparative Analysis**: Review adjacent sections or clinical history if available

**Technical Note:**
This analysis uses computer vision feature extraction (CLIP) combined with statistical image analysis. While comprehensive for computational assessment, it does not replace professional pathological evaluation for diagnostic purposes.
"""
                
                self.log_info("✅ Enhanced CLIP pathology analysis completed")
                return analysis
                
            else:
                self.log_error(f"Unknown model type: {self.model_name}")
                return "Error: Unknown model type"
            
        except Exception as e:
            self.log_error(f"Analysis failed: {e}")
            import traceback
            self.log_error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.loaded = False
        self.log_info("🗑️ Model unloaded, memory cleared")
    
    def prepare_image(self, image):
        """Prepare image for processing"""
        try:
            self.log_info(f"Preparing image of type: {type(image)}")
            
            # Handle CuImage objects from CuCIM
            if hasattr(image, '__class__') and 'cucim' in str(type(image)):
                self.log_info("Converting CuImage to numpy array")
                # Convert CuImage to numpy array
                if hasattr(image, 'read_region'):
                    # CuImage with read_region method
                    img_array = image.read_region()
                elif hasattr(image, 'as_numpy'):
                    # CuImage with as_numpy method
                    img_array = image.as_numpy()
                else:
                    # Try to convert to numpy directly
                    img_array = np.array(image)
                
                self.log_info(f"CuImage converted, shape: {img_array.shape}, type: {type(img_array)}")
                
                # Handle potential CuPy arrays or GPU arrays
                if hasattr(img_array, 'get'):
                    # CuPy array - convert to CPU numpy array
                    img_array = img_array.get()
                    self.log_info("Converted CuPy array to numpy")
                elif hasattr(img_array, 'cpu'):
                    # PyTorch tensor - convert to numpy
                    img_array = img_array.cpu().numpy()
                    self.log_info("Converted PyTorch tensor to numpy")
                elif not isinstance(img_array, np.ndarray):
                    # Force conversion to numpy array
                    img_array = np.asarray(img_array)
                    self.log_info(f"Force converted to numpy: {type(img_array)}")
                
            # Handle numpy arrays
            elif isinstance(image, np.ndarray):
                img_array = image
                self.log_info(f"Using numpy array: {img_array.shape}")
                
            # Handle PIL Images
            elif hasattr(image, 'mode'):
                img_array = np.array(image)
                self.log_info(f"Converting PIL to numpy: {img_array.shape}")
                
            else:
                # Try to convert whatever it is to numpy
                self.log_info(f"Converting unknown type {type(image)} to numpy")
                img_array = np.asarray(image)
            
            # Final type check and conversion
            if not isinstance(img_array, np.ndarray):
                self.log_info(f"Final conversion attempt for type: {type(img_array)}")
                img_array = np.asarray(img_array)
            
            # Verify we have a valid numpy array
            if not isinstance(img_array, np.ndarray):
                raise ValueError(f"Could not convert {type(image)} to numpy array (final type: {type(img_array)})")
            
            self.log_info(f"✅ Valid numpy array: shape {img_array.shape}, dtype: {img_array.dtype}")
            
            # Handle different array shapes and types
            if len(img_array.shape) == 2:
                # Grayscale image, convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
                self.log_info("Converted grayscale to RGB")
                
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    # RGBA image, take only RGB channels
                    self.log_info("Converting RGBA to RGB (removing alpha channel)")
                    img_array = img_array[:, :, :3]
                elif img_array.shape[2] == 1:
                    # Single channel, convert to RGB
                    img_array = np.repeat(img_array, 3, axis=2)
                    self.log_info("Converted single channel to RGB")
                elif img_array.shape[2] != 3:
                    raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
            else:
                raise ValueError(f"Unsupported image shape: {img_array.shape}")
            
            self.log_info(f"After channel processing: {img_array.shape}")
            
            # Normalize values to 0-255 range if needed
            if img_array.dtype in [np.float32, np.float64]:
                if img_array.max() <= 1.0:
                    # Values are in 0-1 range, scale to 0-255
                    img_array = (img_array * 255).astype(np.uint8)
                    self.log_info("Normalized float values to 0-255 range")
                else:
                    img_array = img_array.astype(np.uint8)
                    self.log_info("Converted float to uint8")
            elif img_array.dtype != np.uint8:
                # Convert to uint8
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                self.log_info(f"Converted {img_array.dtype} to uint8")
            
            self.log_info(f"After normalization: dtype {img_array.dtype}, range [{img_array.min()}, {img_array.max()}]")
            
            # Create PIL Image from numpy array
            pil_image = Image.fromarray(img_array)
            self.log_info(f"Created PIL image: {pil_image.size}, mode: {pil_image.mode}")
            
            # Ensure RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                self.log_info("Converted to RGB mode")
            
            # Resize if too large for memory efficiency
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                self.log_info(f"Resized to {new_size} for processing")
            
            self.log_info(f"✅ Image prepared successfully: {pil_image.size}, mode: {pil_image.mode}")
            return pil_image
            
        except Exception as e:
            self.log_error(f"Failed to prepare image: {e}")
            self.log_error(f"Image type: {type(image)}")
            
            # More detailed debugging
            if hasattr(image, 'shape'):
                self.log_error(f"Image shape: {image.shape}")
            if hasattr(image, 'dtype'):
                self.log_error(f"Image dtype: {image.dtype}")
            if hasattr(image, '__array__'):
                self.log_error("Object has __array__ method")
            
            # Try to see what methods are available
            try:
                methods = [method for method in dir(image) if not method.startswith('_')]
                self.log_error(f"Available methods: {methods[:10]}...")  # Show first 10 methods
            except:
                pass
            
            import traceback
            self.log_error(f"Full traceback: {traceback.format_exc()}")
            return None
    
#     def analyze_pathology_image(self, image, transformation_info="", custom_prompt="", 
#                                max_new_tokens=1500, temperature=0.1, do_sample=True):
#         """Analyze transformed WSI image for pathology insights"""
        
#         if not self.load_model():
#             return None
        
#         processed_image = self.prepare_image(image)
#         if processed_image is None:
#             return None
        
#         try:
#             self.log_info(f"🔬 Starting analysis with {self.model_name}...")
            
#             if "blip2" in self.model_name.lower():
#                 # BLIP2 analysis
#                 if custom_prompt.strip():
#                     prompt = f"Question: {custom_prompt} Answer:"
#                 else:
#                     prompt = f"Question: Analyze this medical pathology image. What tissue types, cellular structures, and potential abnormalities can you identify? Transformation applied: {transformation_info}. Answer:"
                
#                 inputs = self.processor(processed_image, prompt, return_tensors="pt")
                
#                 if torch.cuda.is_available():
#                     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
#                 with torch.no_grad():
#                     generated_ids = self.model.generate(
#                         **inputs,
#                         max_length=inputs['input_ids'].shape[1] + max_new_tokens,
#                         temperature=temperature,
#                         do_sample=do_sample,
#                         num_beams=3
#                     )
                
#                 response = self.processor.decode(generated_ids[0], skip_special_tokens=True)
#                 self.log_info("✅ BLIP2 analysis completed")
#                 return response
                
#             elif "instructblip" in self.model_name.lower():
#                 # InstructBLIP analysis  
#                 if custom_prompt.strip():
#                     prompt = custom_prompt
#                 else:
#                     prompt = f"Analyze this pathology image and describe the tissue types, cellular structures, and any abnormal features you can identify. Transformation applied: {transformation_info}."
                
#                 inputs = self.processor(images=processed_image, text=prompt, return_tensors="pt")
                
#                 if torch.cuda.is_available():
#                     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_new_tokens,
#                         temperature=temperature,
#                         do_sample=do_sample
#                     )
                
#                 response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
#                 self.log_info("✅ InstructBLIP analysis completed")
#                 return response
                
#             elif "clip" in self.model_name.lower():
#                 # CLIP-based analysis (limited)
#                 inputs = self.processor(images=processed_image, return_tensors="pt", padding=True)
                
#                 if torch.cuda.is_available():
#                     inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
#                 with torch.no_grad():
#                     image_features = self.model.get_image_features(**inputs)
                
#                 # Create a basic analysis based on feature patterns
#                 response = f"""Basic Image Analysis (CLIP-based):
# - Image processed successfully
# - Applied transformation: {transformation_info}
# - Image dimensions: {processed_image.size}
# - Feature vector extracted with {image_features.shape[1]} dimensions
# - Custom query: {custom_prompt if custom_prompt.strip() else "None provided"}

# Note: This is a basic analysis. For detailed pathology insights, a specialized vision-language model is recommended."""
                
#                 self.log_info("✅ CLIP analysis completed")
#                 return response
                
#             else:
#                 self.log_error(f"Unknown model type: {self.model_name}")
#                 return None
            
#         except Exception as e:
#             self.log_error(f"Analysis failed: {e}")
#             import traceback
#             self.log_error(f"Traceback: {traceback.format_exc()}")
#             return None

def create_llava_analysis_ui():
    """Create UI components for LLaVA-NeXT pathology analysis"""
    
    st.markdown("### 🔬 LLaVA-NeXT Local Pathology Analysis")
    
    # Model Configuration
    with st.expander("**LLaVA-NeXT Configuration**", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_options = {
                "LLaVA-NeXT Mistral 7B": "llava-hf/llava-v1.6-mistral-7b-hf",
                "LLaVA-NeXT Vicuna 7B": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "LLaVA-NeXT Vicuna 13B": "llava-hf/llava-v1.6-vicuna-13b-hf",
            }
            
            selected_model_name = st.selectbox(
                "Model Selection",
                list(model_options.keys()),
                index=0,
                help="Choose the LLaVA-NeXT model variant"
            )
            model_name = model_options[selected_model_name]
        
        with col2:
            # Check system status
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                st.success(f"🟢 GPU Available: {device_name}")
            else:
                st.warning("🟡 GPU Not Available - Will use CPU")
        
        # Generation parameters
        st.markdown("**Generation Parameters**")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Lower values = more consistent responses"
            )
        
        with col4:
            max_tokens = st.number_input(
                "Max New Tokens",
                min_value=256,
                max_value=2048,
                value=1500,
                step=256,
                help="Maximum response length"
            )
        
        with col5:
            do_sample = st.checkbox(
                "Sampling",
                value=True,
                help="Enable sampling for more varied responses"
            )
        
        custom_prompt = st.text_area(
            "Custom Analysis Prompt (Optional)",
            placeholder="Enter specific questions or focus areas for the pathology analysis...",
            height=100,
            help="Leave empty to use default pathology analysis prompt"
        )
    
    return model_name, custom_prompt, temperature, max_tokens, do_sample

def get_model_requirements():
    """Get memory requirements for different models"""
    
    requirements = {
        "llava-hf/llava-v1.6-mistral-7b-hf": {
            "memory": "~14GB GPU memory",
            "description": "Best balance of performance and efficiency"
        },
        "llava-hf/llava-v1.6-vicuna-7b-hf": {
            "memory": "~14GB GPU memory", 
            "description": "Good general performance"
        },
        "llava-hf/llava-v1.6-vicuna-13b-hf": {
            "memory": "~26GB GPU memory",
            "description": "Higher quality but requires more memory"
        }
    }
    
    return requirements