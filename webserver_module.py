import http.server
import socketserver
from io import BytesIO
from urllib.parse import urlparse, parse_qs
import torch
from diffusers import AutoPipelineForText2Image
from tensorflow.keras.models import load_model
from compel import Compel, ReturnedEmbeddingsType
from sampling_module import generate_text
from vectorizer import load_vectorizer


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global g_model
        global g_vectorizer
        global g_temperature
        global g_max_output_tokens
        global g_pipe
        global g_compel

        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)
        print(parsed_path)
        print(params)

        if parsed_path.path == '/card-data.js':
            self.send_response(200)
            self.send_header('Content-type', 'text/javascript')
            self.end_headers()

            generated_text = generate_text(
                g_max_output_tokens,
                g_model,
                g_vectorizer,
                temperature=g_temperature,
                show_token_breaks=False
            )

            self.wfile.write(f'export default\n{generated_text}'.encode())
        elif parsed_path.path == '/card-image.png':
            self.send_response(200)
            self.send_header('Content-type', 'image')
            self.end_headers()

            prompt_items = [f'({params["name"][0]})+++']
            if "type[]" in params:
                prompt_items.extend([
                    f'({type})'
                    for type in params["type[]"]
                ])
            if "subtype[]" in params:
                prompt_items.extend([
                    f'({subtype})+++'
                    for subtype in params["subtype[]"]
                ])
            if "supertype[]" in params:
                prompt_items.extend([
                    f'({supertype})'
                    for supertype in params["supertype[]"]
                ])
            prompt_items.extend([
                "(art by Alayna Lemmer)",
                "(art by Alex Horley)",
                "(art by Even Mehl Amundsen)"
            ])

            prompt = ", ".join(prompt_items)
            print(prompt)
            conditioning, pooled = g_compel(prompt)

            image = g_pipe(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                num_inference_steps=2,
                guidance_scale=0.0,
                width=768,
                height=512
            ).images[0]

            with BytesIO() as buffer:
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

            self.wfile.write(image_bytes)

        elif parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            with open('./html/card.html', 'r') as file:
                self.wfile.write(file)
        else:
            # Call the parent class method to handle other requests
            super().do_GET()


def start_server(
        model_path=None,
        temperature=None,
        vectorizer_path=None,
        max_output_tokens=None,
        listen_address=None,
        port=None
):
    global g_model
    global g_vectorizer
    global g_temperature
    global g_max_output_tokens
    global g_pipe
    global g_compel

    g_vectorizer = load_vectorizer(vectorizer_path)
    g_model = load_model(model_path)

    g_temperature = temperature
    g_max_output_tokens = max_output_tokens

    # Diffusers
    g_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    g_pipe.to("cuda")

    g_compel = Compel(
        tokenizer=[g_pipe.tokenizer, g_pipe.tokenizer_2],
        text_encoder=[g_pipe.text_encoder, g_pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        device='cuda'
    )

    # Specify the custom handler
    handler = MyHandler

    # Create the server
    with socketserver.TCPServer((listen_address, port), handler) as httpd:
        print(f"Serving on {listen_address}:{port}")
        httpd.serve_forever()
