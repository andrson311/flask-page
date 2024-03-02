import os
import json
from dotenv import load_dotenv
from pprint import pprint
from flask import Flask, render_template
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import io
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

load_dotenv()
ROOT = os.path.dirname(__file__)
MENU_IMAGES_DIR = os.path.join('img', 'menu')
STABILITY_KEY = os.getenv('STABILITY_KEY')
DATA = os.path.join(ROOT, 'data.json')
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = STABILITY_KEY

def load_data():
    try:
        with open(DATA, 'r') as data_file:
            data = json.load(data_file)
            return data
    except:
        return
    
def save_data(data):
    with open(DATA, 'w') as data_file:
        json.dump(data, data_file, indent=4)

def generate_menu_items(menu_type):
    response_schemas = [
        ResponseSchema(
            name='dish titles', 
            description='List of 6 dishes that are made out of potatoes'
        ),
        ResponseSchema(
            name='ingredients',
            description='Python list of 5 main ingredients for each dish'
        ),
        ResponseSchema(
            name='image prompts',
            description='List of prompts for image generating model for dishes'
        ),
        ResponseSchema(
            name='prices',
            description='List of dish prices in US dollars'
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template='Give me information about the following dishes that we could put on a {menu_type} menu\n{format_instructions}',
        input_variables=['menu_type'],
        partial_variables={'format_instructions': format_instructions}
    )

    model = ChatOpenAI(temperature=0)
    chain = prompt | model | output_parser

    output = chain.invoke({'menu_type': menu_type})
    return output

def generate_menu_item_image(prompt, img_title):
    stability_api = client.StabilityInference(
        key=STABILITY_KEY,
        verbose=False,
        engine='stable-diffusion-xl-1024-v1-0'
    )

    output = stability_api.generate(prompt=prompt)
    output_path = os.path.join(MENU_IMAGES_DIR, img_title + '.png')

    for r in output:
        for artifact in r.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(os.path.join(ROOT, 'static', output_path))
    
    return output_path

def get_menu_data():
    menu = {}
    menu_types = [
        'starters',
        'breakfast',
        'lunch',
        'dinner'
    ]

    data = load_data()
    if data and 'menu' in data:
        menu = data['menu']
    else:
        for m in menu_types:
            output = generate_menu_items(m)
            if len(output['dish titles']) == len(output['ingredients']):
                menu['menu-' + m] = {}
                menu['menu-' + m]['name'] = m.capitalize()
                menu['menu-' + m]['data'] = []

                for d in range(len(output['dish titles'])):
                    dish_title = output['dish titles'][d]
                    img_path = generate_menu_item_image(output['image prompts'][d], f"{m}-{d}")
                    menu['menu-' + m]['data'].append({
                        'dish': dish_title,
                        'ingredients': ', '.join(output['ingredients'][d]),
                        'img': img_path,
                        'price': output['prices'][d]
                    })

        if data:
            data['menu'] = menu
        else:
            data = {
                'menu': menu
            }
        save_data(data)

    return menu

app = Flask(__name__)

@app.route('/')
def home():
    menu = get_menu_data()
    return render_template('index.html', menu=menu)
