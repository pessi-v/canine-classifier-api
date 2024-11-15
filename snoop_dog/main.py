from snoop_dog.predict import format_predictions
from snoop_dog.class_names import class_names
from snoop_dog.registry import load_and_resize_image, dogimg_process, gradcam
    

# put it all together for the api
def predict_breeds(model, img):
    image = load_and_resize_image(img)
    image_processed = dogimg_process(image)
    model = model
    predictions = model.predict(image_processed)
    output = format_predictions(predictions, class_names)
    output[0]['gradcam'] = gradcam(model, image_processed)
    return output
