import streamlit as st
import pickle


Forest_model = pickle.load(open('Forest_model.pkl','rb'))
XG_model = pickle.load(open('XG_model.pkl','rb'))




def main():
    st.title("Melbourne Hosue Prices")
    html_temp = """
    <div style="background-color:purple;padding:5px">
    <h2 style="color:pink;text-align:center;">Predict the House Price</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Random Forest Regressor']
    option=st.sidebar.selectbox('Choose the model',activities)
    st.subheader(option)
    room=st.slider('Rooms', 0, 10)
    # type = ["House", "Unit", "Town House"]
    # type_selected_option = st.selectbox("Select an option:", type)
    dis=st.slider('Distance', 0.0, 50.0)
    post=st.number_input("Enter Postal Code:", min_value=3000, max_value=4000)
    bed=st.slider('Bedrooms', 0, 10)
    bath=st.slider('Bathrooms', 0, 8)
    c=st.slider('Car Spots', 0, 10)
    land=st.slider('Land Size', 0.0, 450000.0)
    year=st.number_input("Enter the Year the House was built:" ,min_value=1100, max_value=2020 )
    lat =   st.slider('Latiude ', -39.0, -37.0)
    long =   st.slider('Longitude ', 144.0, 146.0)
   
    inputs=[[room,dis,post,bed,bath,c,land,year,lat , long]]
    if st.button('Predict'):
        if option=='Random Forest Regressor':
            st.success("The estimated cost is " + str(Forest_model.predict(inputs)[0])+" $")
        else:
           #st.success((XG_model.predict(inputs)))
           return 0


if __name__=='__main__':
    main()
