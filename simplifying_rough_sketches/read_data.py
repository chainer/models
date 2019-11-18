import os

def get_data(Input, Target):
    Targets =sorted(os.listdir(Target))
    Inputs = sorted(os.listdir(Input))
    input_images, target_images = [], []
    for input_image, target_image in zip(Inputs, Targets):
        image_path = os.path.join(Input, input_image)
        target_path = os.path.join(Target, target_image)

        input_images.append(image_path)
        target_images.append(target_path)

    return input_images, target_images
