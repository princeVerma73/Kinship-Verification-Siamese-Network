import os
import tempfile

import streamlit as st
from PIL import Image

from predict import predict

# ------------------------------------
# Page Config
# ------------------------------------

st.set_page_config(
    page_title="Kinship Verification",
    page_icon="👨‍👩‍👧",
    layout="wide"
)

# ------------------------------------
# Title
# ------------------------------------

st.title("👨‍👩‍👧 Kinship Verification")

st.write(
    "Upload two face images and the model will predict whether they belong to related family members."
)

# Upload Images


col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader(
        "Upload First Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    img2 = st.file_uploader(
        "Upload Second Image",
        type=["jpg", "jpeg", "png"]
    )


# Preview Images

if img1 and img2:

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            Image.open(img1),
            caption="First Image",
            width=300
        )

    with col2:
        st.image(
            Image.open(img2),
            caption="Second Image",
            width=300
        )

# ------------------------------------
# Predict
# ------------------------------------

if st.button("Predict"):

    if img1 is None or img2 is None:
        st.warning("Please upload both images.")

    else:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1:
            temp1.write(img1.getvalue())
            path1 = temp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
            temp2.write(img2.getvalue())
            path2 = temp2.name

        try:

            with st.spinner("Predicting..."):

                result = predict(path1, path2)

            st.success("Prediction Completed")

            st.markdown("---")

            st.subheader("Result")

            if result["prediction"] == "Related":

                st.success(
                    f"✅ Related\n\nConfidence : {result['confidence']}%"
                )

            else:

                st.error(
                    f"❌ Not Related\n\nConfidence : {result['confidence']}%"
                )

        finally:

            if os.path.exists(path1):
                os.remove(path1)

            if os.path.exists(path2):
                os.remove(path2)