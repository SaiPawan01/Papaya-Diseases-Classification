import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import gradio as gr


device = 'cuda' if torch.cuda.is_available() else "cpu"


model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(1024, 8),        
)


model.load_state_dict(torch.load('models/densenet_new_10_model.pth',weights_only=True,map_location=torch.device(device)))
model.to(device)
model.eval()


# transform
transform = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor()
])

# Prediction function
def predict_tumor(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)

    # probs = torch.softmax(outputs,1)[0]
    _, predicted = torch.max(outputs, 1)
    
    class_names = ['Anthracnose',
                    'BacterialSpot',
                    'Curl',
                    'Healthy',
                    'Mealybug',
                    'Mite disease',
                    'Mosaic',
                    'Ringspot']
    
    
    return f"Disease : {class_names[predicted.item()]}"



# Custom CSS styling
css = """
body {
  background-color: #0D1117;
  color: #E6EDF3;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  margin: 0;
  padding: 0;
}

h1 {
  font-size: 36px;
  text-align: center;
  margin-bottom: 20px;
  color: #58A6FF;
}

footer {
  text-align: center;
  color: #8B949E;
  padding: 20px 0;
  font-size: 14px;
}
"""

# Gradio Interface
app = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=8, label="Prediction"),
    title="Papaya Leaf Disease Prediction",
    description="Upload a papaya leaf image — this model will predict disease.",
    theme="soft",
    css=css,
)

# Launch the app
app.launch()