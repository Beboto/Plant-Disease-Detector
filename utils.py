from torchvision import transforms
import torch

# Your class names — make sure this is the correct list and in the correct order
CLASS_NAMES = [
    'Millet_blast',
    'Millet_Healthy',
    'Millet_rust',
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Late_blight',
    'Potato___Early_blight',
    'Potato___healthy',
    'Rice_bacterial_leaf_blight',
    'Rice_brown_spot',
    'Rice_healthy',
    'Rice_leaf_blast',
    'Rice_leaf_scald',
    'Rice_narrow_brown_spot',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato___healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus'
]

# Use the *same transform as training*
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(model, img):
    img_t = transform(img).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        return CLASS_NAMES[predicted.item()]
