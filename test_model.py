from predict import predict

test_cases = [

    # Related
    (
        "train-faces/F0002/MID1/P00009_face3.jpg",
        "train-faces/F0002/MID2/P00009_face2.jpg"
    ),

    # Unrelated (change to real paths that exist)
    (
        "train-faces/F0002/MID1/P00009_face3.jpg",
        "train-faces/F0010/MID1/P00123_face1.jpg"
    )

]

for img1, img2 in test_cases:

    result = predict(img1, img2)

    print("=" * 60)
    print(img1)
    print(img2)
    print(result)