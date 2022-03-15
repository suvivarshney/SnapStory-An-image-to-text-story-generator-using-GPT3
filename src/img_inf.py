
from img2_caption import *

image_path = r'test_img.jpeg' # enter your own path

#load models
image_features_extract_model, tokenizer, encoder, decoder = load_models()

#predict using image path
result, attention_plot = predict(image_path,image_features_extract_model, tokenizer, encoder, decoder)
caption = ' '.join(result)

print ('Prediction Caption:', caption)