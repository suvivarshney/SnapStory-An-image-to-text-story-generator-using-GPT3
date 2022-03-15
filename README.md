# SnapStory-An-image-to-text-story-generator-using-GPT3
AI based story generation for ECS289G at UC Davis. 

GPT3 from OPENAI requires a key to use, which is billable afte a few hits. We cannot add our key to a public repository. To add the key, got to src/gpt3 and in the line 15, in the function "def gpt3_init():" add your own key for GPT3.

To run the model, got to src/ and do 'flask run'. The front-end should be uo at the mentioned url, which by default is '127.0.0.1:5000'
