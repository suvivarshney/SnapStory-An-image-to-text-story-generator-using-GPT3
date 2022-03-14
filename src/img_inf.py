'''
-------------------------------------
Files required for img2_caption model
-------------------------------------

//tree structure
    img2_caption.py
    tokenizer.pickle
    checkpoints
    |---train
        |---checkpoint
        |---ckpt-4.data-00000-of-00001
        |---ckpt-4.index

-------------
Model outputs
-------------
result: list of tokens predicted from image  eg.['words', 'in', 'caption', '<end>']
attention_plot: pls ignore

***caption: ready to use sentence eg. "beautiful multi colored sheep on the grass in an open grassy area <end>"
-------------

Please use code below to access img2_caption model:
'''
from img2_caption import *

image_path = r'test_img_2.jpeg' # enter your own path

#load models
image_features_extract_model, tokenizer, encoder, decoder = load_models()

#predict using image path
result, attention_plot = predict(image_path,image_features_extract_model, tokenizer, encoder, decoder)
caption = ' '.join(result)

print ('Prediction Caption:', caption)