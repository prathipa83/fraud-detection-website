import numpy as np
import pickle
import time
import streamlit as st
from PIL import Image

# Load the saved model safely
try:
    with open('final_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Creating a function for prediction with caching
@st.cache(allow_output_mutation=True)
def predict_fraud(card1: float, card2: float, card4: int, card6: int,
                  addr1: int, addr2: int, TransactionAmt: float,
                  P_emaildomain: int, ProductCD: int, DeviceType: int) -> float:
    input_array = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt,
                             P_emaildomain, ProductCD, DeviceType]])
    prediction = loaded_model.predict_proba(input_array)
    pred = float("{0:.2f}".format(prediction[0][0]))
    return pred

def main():
    st.markdown(
        """
        <div style="background-color:#000000; padding:10px">
            <h1 style="color:white; text-align:center;">
                Financial Transaction Fraud Prediction ML Web App 💰
            </h1>
        </div>
        """, unsafe_allow_html=True
    )

    try:
        image = Image.open('home_banner.PNG')
        st.image(image, caption='Impacting the World of Finance and Banking with Artificial Intelligence (AI)')
    except Exception as e:
        st.error(f"Error loading image: {e}")

    st.sidebar.title("Financial Transaction Fraud Prediction System 🕵️")
    st.sidebar.subheader("Choose the Below Parameters to Predict a Transaction")

    TransactionAmt = st.sidebar.number_input("Transaction Amount (USD)", 0, 20000, step=1)
    card1 = st.sidebar.number_input("Payment Card 1 Amount (USD)", 0, 20000, step=1)
    card2 = st.sidebar.number_input("Payment Card 2 Amount (USD)", 0, 20000, step=1)
    card4 = st.sidebar.radio("Payment Card Category", [1, 2, 3, 4])
    st.sidebar.info("1: Discover | 2: Mastercard | 3: American Express | 4: Visa")
    card6 = st.sidebar.radio("Payment Card Type", [1, 2])
    st.sidebar.info("1: Credit | 2: Debit")
    addr1 = st.sidebar.slider("Billing Zip Code", 0, 500, step=1)
    addr2 = st.sidebar.slider("Billing Country Code", 0, 100, step=1)
    P_emaildomain = st.sidebar.selectbox("Purchaser Email Domain", [0, 1, 2, 3, 4])
    st.sidebar.info("0: Gmail (Google) | 1: Outlook (Microsoft) | 2: Mail.com | 3: Others | 4: Yahoo")
    ProductCD = st.sidebar.selectbox("Product Code", [0, 1, 2, 3, 4])
    st.sidebar.info("0: C | 1: H | 2: R | 3: S | 4: W")
    DeviceType = st.sidebar.radio("Payment Device Type", [1, 2])
    st.sidebar.info("1: Mobile | 2: Desktop")

    safe_html = """ 
    <img src="https://media.giphy.com/media/g9582DNuQppxC/giphy.gif" alt="confirmed" style="width:698px;height:350px;"> 
    """
    danger_html = """  
    <img src="https://media.giphy.com/media/8ymvg6pl1Lzy0/giphy.gif" alt="cancel" style="width:698px;height:350px;">
    """

    if st.button("Click Here To Predict"):
        output = predict_fraud(card1, card2, card4, card6, addr1, addr2,
                               TransactionAmt, P_emaildomain, ProductCD, DeviceType)
        final_output = output * 100
        st.subheader(f'Probability Score of Financial Transaction is {final_output:.2f}%')

        if final_output > 75.0:
            st.markdown(danger_html, unsafe_allow_html=True)
            st.error("**OMG! Financial Transaction is Fraudulent**")
        else:
            st.balloons()
            time.sleep(5)
            st.balloons()
            time.sleep(5)
            st.balloons()
            st.markdown(safe_html, unsafe_allow_html=True)
            st.success("**Hurray! Transaction is Legitimate**")

if __name__ == '__main__':
    main()



