import os
import json
import base64
import requests
from together import Together
from PIL import Image
import io
import tempfile
import webbrowser
from dotenv import load_dotenv
import socket
import time

# Load environment variables
load_dotenv()

def generate_image_prompts(concept, num_prompts=3):
    """
    Use the Nillion API to generate detailed image prompts from a simple concept
    
    Args:
        concept (str): Simple concept like "on beach"
        num_prompts (int): Number of different prompts to generate
        
    Returns:
        list: List of generated detailed prompts
    """
    try:
        # Get environment variables for Nillion API
        api_url = os.environ.get("NILAI_API_URL", "https://nilai-a779.nillion.network")
        api_key = os.environ.get("NILAI_API_KEY", "Nillion2025")
        
        # Prepare system message to instruct LLM to generate image prompts
        system_message = """
        You are a specialized image prompt generator. 
        Create detailed, high-quality image prompts for the concept provided.
        Each prompt should be different but related to the same concept.
        Make the prompts specific, visual, and detailed enough for an image generation model.
        ONLY return the prompts, one per line, with no additional explanation or commentary.
        Prompt MUST start with 'n3lson man', eg: 'n3lson man on beach' since 'n3lson' is the keyword for the image fine tuned model.
        """
        
        # Prepare the request
        url = f"{api_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate {num_prompts} different detailed image prompts for the concept: '{concept}'"}
            ],
            "temperature": 0.7,  # Slightly higher temperature for creativity
        }
        
        # Make the API call
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        generated_text = result['choices'][0]['message']['content']
        
        # Split text into individual prompts (one per line)
        prompts = [p.strip() for p in generated_text.strip().split('\n') if p.strip()]
        
        # Limit to requested number of prompts
        prompts = prompts[:num_prompts]
        
        print(f"Generated {len(prompts)} image prompts:")
        for i, prompt in enumerate(prompts):
            print(f"{i+1}. {prompt}")
            
        return prompts
        
    except Exception as error:
        print(f"Error generating prompts: {error}")
        # Fallback prompts if API call fails
        return [f"n3lson {concept} realistic photo high definition" for _ in range(num_prompts)]


class ImageGenerator:
    def __init__(self):
        # Load environment variables if not already loaded
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is required. Add it to .env file.")
            
        self.client = Together(api_key=self.api_key)
        self.generated_images = []
        self.prompts_used = []
        
    def generate_images_from_prompts(self, prompts, 
                               model="black-forest-labs/FLUX.1-dev-lora",
                               width=1024, 
                               height=768, 
                               steps=28,
                               lora_path="http://hills.ccsf.edu/~clai74/nelson_unet.safetensors",
                               lora_scale=1.0):
        """
        Generate one image for each provided prompt
        
        Args:
            prompts (list): List of text prompts to generate images from
            model (str): Model name to use
            width (int): Image width
            height (int): Image height
            steps (int): Number of inference steps
            lora_path (str): Path to LoRA adapter
            lora_scale (float): Scale factor for LoRA adapter
            
        Returns:
            list: List of base64 encoded images
        """
        images = []
        self.prompts_used = []
        
        for i, prompt in enumerate(prompts):
            try:
                print(f"Generating image {i+1}/{len(prompts)}...")
                response = self.client.images.generate(
                    prompt=prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    n=1,
                    response_format="b64_json",
                    image_loras=[{"path": lora_path, "scale": lora_scale}]
                )
                
                # Store the generated image
                if response.data and len(response.data) > 0:
                    images.append(response.data[0].b64_json)
                    self.prompts_used.append(prompt)
                    print(f"Image {i+1} generated successfully")
                else:
                    print(f"No image data returned for prompt {i+1}")
                    
            except Exception as e:
                print(f"Error generating image {i+1}: {e}")
                
        # Store all generated images
        self.generated_images = images
        
        return images
    
    def display_images_for_selection(self, concept):
        """
        Display the generated images in a simple HTML page for selection
        
        Returns:
            str: Path to the HTML file
        """
        if not self.generated_images:
            print("No images to display")
            return None
            
        # Create HTML page with images
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Select an Image</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { text-align: center; }
                .image-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px; }
                .image-option { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; width: 45%; max-width: 600px; margin-bottom: 20px; }
                .image-option img { width: 100%; height: auto; display: block; }
                .select-btn { background: #4CAF50; color: white; border: none; padding: 10px; width: 100%; cursor: pointer; font-size: 16px; }
                .select-btn:hover { background: #45a049; }
                .story-btn { background: #2196F3; color: white; border: none; padding: 10px; width: 100%; cursor: pointer; font-size: 16px; margin-top: 5px; }
                .story-btn:hover { background: #0b7dda; }
                .image-info { padding: 10px; background: #f9f9f9; }
                .prompt-text { font-size: 12px; color: #666; margin: 5px 0; }
                .form-group { margin-top: 10px; }
                .form-group label { display: block; margin-bottom: 5px; }
                .form-group input, .form-group textarea { width: 100%; padding: 8px; box-sizing: border-box; }
                .modal { display: none; position: fixed; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }
                .modal-content { background-color: #fefefe; margin: 15% auto; padding: 20px; border: 1px solid #888; width: 80%; max-width: 500px; border-radius: 8px; }
                .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }
                .close:hover { color: black; }
                .success-message { color: green; display: none; margin-top: 10px; }
                .error-message { color: red; display: none; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Select an Image</h1>
            <div class="image-container">
        """
        
        for i, (img_data, prompt) in enumerate(zip(self.generated_images, self.prompts_used)):
            img_src = f"data:image/png;base64,{img_data}"
            html_content += f"""
                <div class="image-option" id="option-{i+1}">
                    <img src="{img_src}" alt="Generated image {i+1}">
                    <div class="image-info">
                        <h3>Option {i+1}</h3>
                        <div class="prompt-text">Prompt: {prompt}</div>
                        <button class="select-btn" onclick="selectImage({i})">Select This Image</button>
                        <button class="story-btn" onclick="openStoryModal({i}, '{prompt.replace("'", "&apos;")}')">Post to Story Protocol</button>
                    </div>
                </div>
            """
        
        # Add Story Protocol Modal
        html_content += f"""
            </div>
            
            <!-- Story Protocol Modal -->
            <div id="storyModal" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal()">&times;</span>
                    <h2>Post to Story Protocol</h2>
                    <div class="form-group">
                        <label for="title">Title:</label>
                        <input type="text" id="title" value="AI Generated: {concept}" />
                    </div>
                    <div class="form-group">
                        <label for="description">Description:</label>
                        <textarea id="description" rows="4">AI-generated image created from the concept: '{concept}'</textarea>
                    </div>
                    <button class="story-btn" onclick="submitToStoryProtocol()">Submit</button>
                    <div id="successMessage" class="success-message">Successfully posted to Story Protocol!</div>
                    <div id="errorMessage" class="error-message">Error posting to Story Protocol. Please try again.</div>
                </div>
            </div>
            
            <script>
                let selectedImageIndex = -1;
                let selectedPrompt = '';
                
                function selectImage(index) {{
                    const selectedFile = `selected_image_${{index}}.png`;
                    console.log("Selected image:", index);
                    alert(`You selected image ${{index + 1}}! In a complete application, this would be saved as ${{selectedFile}}`);
                    // Here you would typically communicate back to Python
                    fetch('http://localhost:5001/select-image', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ index: index }})
                    }});
                }}
                
                function openStoryModal(index, prompt) {{
                    selectedImageIndex = index;
                    selectedPrompt = prompt;
                    document.getElementById('storyModal').style.display = 'block';
                }}
                
                function closeModal() {{
                    document.getElementById('storyModal').style.display = 'none';
                    document.getElementById('successMessage').style.display = 'none';
                    document.getElementById('errorMessage').style.display = 'none';
                }}
                
                function submitToStoryProtocol() {{
                    if (selectedImageIndex < 0) return;
                    
                    const title = document.getElementById('title').value;
                    const description = document.getElementById('description').value;
                    
                    // Get image data
                    const imgElement = document.querySelector(`#option-${{selectedImageIndex + 1}} img`);
                    const imageData = imgElement.src.split(',')[1]; // Get the base64 part without the prefix
                    
                    console.log("Submitting to Story Protocol with title: " + title);
                    
                    // Send to Story Protocol integration service
                    fetch('http://localhost:3000/register-image', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            imageBase64: imageData,
                            title: title,
                            description: description,
                            prompt: selectedPrompt
                        }})
                    }})
                    .then(response => {{
                        if (!response.ok) {{
                            throw new Error('Network response was not ok: ' + response.status);
                        }}
                        return response.json();
                    }})
                    .then(data => {{
                        if (data.success) {{
                            document.getElementById('successMessage').style.display = 'block';
                            document.getElementById('errorMessage').style.display = 'none';
                            console.log('Posted to Story Protocol:', data);
                        }} else {{
                            document.getElementById('errorMessage').style.display = 'block';
                            document.getElementById('successMessage').style.display = 'none';
                            document.getElementById('errorMessage').textContent = 'Error: ' + (data.error || 'Unknown error');
                            console.error('Error posting to Story Protocol:', data.error);
                        }}
                    }})
                    .catch(error => {{
                        document.getElementById('errorMessage').style.display = 'block';
                        document.getElementById('successMessage').style.display = 'none';
                        document.getElementById('errorMessage').textContent = 'Error: ' + error.message;
                        console.error('Error posting to Story Protocol:', error);
                    }});
                }}
                
                // Close modal when clicking outside of it
                window.onclick = function(event) {{
                    const modal = document.getElementById('storyModal');
                    if (event.target == modal) {{
                        closeModal();
                    }}
                }}
            </script>
        </body>
        </html>
        """
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            f.write(html_content.encode('utf-8'))
            html_path = f.name
            
        # Open in browser
        webbrowser.open('file://' + html_path)
        
        return html_path
    
    def save_image(self, index=0, filename=None):
        """
        Save a specific image to disk
        
        Args:
            index (int): Index of the image to save
            filename (str, optional): Filename to save to. If None, generates one.
            
        Returns:
            str: Path to the saved image
        """
        if not self.generated_images or index >= len(self.generated_images):
            print("Invalid image index")
            return None
            
        if filename is None:
            filename = f"generated_image_{index}.png"
            
        try:
            # Convert base64 to image and save
            image_data = base64.b64decode(self.generated_images[index])
            image = Image.open(io.BytesIO(image_data))
            image.save(filename)
            print(f"Image saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def register_with_story_protocol(self, image_index, title, description, prompt):
        """
        Register a generated image with Story Protocol
        
        Args:
            image_index (int): Index of the image to register
            title (str): Title for the image
            description (str): Description of the image
            prompt (str): The prompt used to generate the image
            
        Returns:
            dict: Registration result information
        """
        if not self.generated_images or image_index >= len(self.generated_images):
            print("Invalid image index")
            return None
            
        try:
            # Prepare data for Story Protocol integration
            image_base64 = self.generated_images[image_index]
            
            # Call Story Protocol integration service
            story_api_url = "http://localhost:3000/register-image"
            response = requests.post(
                story_api_url,
                json={
                    "imageBase64": image_base64,
                    "title": title,
                    "description": description,
                    "prompt": prompt or self.prompts_used[image_index]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Successfully registered with Story Protocol. IP ID: {result.get('storyProtocolId')}")
                return result
            else:
                print(f"Error registering with Story Protocol: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error registering with Story Protocol: {e}")
            return None


def create_images_from_concept(concept, num_variations=3):
    """
    Main function to generate images from a simple concept
    
    1. Convert concept to detailed prompts using LLM
    2. Generate an image for each prompt
    3. Display images for selection
    
    Args:
        concept (str): Simple concept like "on beach"
        num_variations (int): Number of image variations to generate
        
    Returns:
        tuple: (list of generated images, list of prompts used)
    """
    # Step 1: Generate detailed prompts from the concept
    prompts = generate_image_prompts(concept, num_variations)
    
    # Step 2: Generate images from the prompts
    try:
        generator = ImageGenerator()
        images = generator.generate_images_from_prompts(prompts)
        
        # Step 3: Display images for selection
        if images:
            html_path = generator.display_images_for_selection(concept)
            return images, generator.prompts_used
        else:
            print("No images were generated")
            return [], []
            
    except Exception as e:
        import traceback
        print(f"Error in image generation process: {e}")
        print(traceback.format_exc())  # Print the full traceback to identify where the error happens
        return [], []


# Example usage
if __name__ == "__main__":
    import sys
    import threading
    import flask
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    # Create a simple Flask server to handle callbacks from the HTML page
    app = Flask(__name__)
    CORS(app)
    
    # Global variables to store the selected image
    selected_image_index = -1
    
    @app.route('/select-image', methods=['POST'])
    def select_image():
        global selected_image_index
        data = request.json
        selected_image_index = data.get('index', -1)
        print(f"Selected image index: {selected_image_index}")
        return jsonify({"status": "success"})
    
    # Start Flask server in a separate thread
    def start_flask_server():
        app.run(host='0.0.0.0', port=5001)
    
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Function to check if a port is in use
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    # Start the Story Protocol integration service
    def start_story_protocol_service():
        # Check if the service is already running
        if is_port_in_use(3000):
            print("Story Protocol service is already running on port 3000")
            return
            
        print("Starting Story Protocol integration service...")
        os.system("cd story-protocol-integration && node index.js")
    
    story_thread = threading.Thread(target=start_story_protocol_service)
    story_thread.daemon = True
    story_thread.start()
    
    # Give the server a moment to start if it wasn't already running
    time.sleep(1)
    
    # Verify API keys are available
    if not os.environ.get("TOGETHER_API_KEY"):
        print("Error: TOGETHER_API_KEY not found in environment variables.")
        print("Please add it to your .env file in the format: TOGETHER_API_KEY=your_api_key_here")
        sys.exit(1)
        
    # Get concept from command line or use default
    user_concept = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "on beach with sunglasses"
    
    # Number of variations to generate
    num_variations = 3
    
    print(f"Generating {num_variations} images for concept: '{user_concept}'")
    
    # Generate images from concept
    images, prompts = create_images_from_concept(user_concept, num_variations)
    
    print(f"Generated {len(images)} images")
    
    # Keep the main thread running to allow user interaction with the HTML
    try:
        while True:
            # Check if an image was selected
            if selected_image_index >= 0:
                print(f"User selected image {selected_image_index + 1}")
                # Here you could automatically save the selected image
                # Or perform other actions with the selected image
                selected_image_index = -1  # Reset selection
            
            # Sleep to avoid high CPU usage
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting application")
        sys.exit(0)