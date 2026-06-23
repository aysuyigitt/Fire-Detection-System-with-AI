from camera_classifier import predict_image

images = [
    "camera_samples/camera_0.jpg",
    "camera_samples/camera_1.jpg",
    "camera_samples/camera_2.jpg",
]

for img in images:
    result = predict_image(img)
    print(img, "=>", result)