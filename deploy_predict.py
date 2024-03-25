from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from joblib import load
import numpy as np

# Vectorize text using the created vectorizer
def vectorize_text(text):
    vectorizer = load('./weights/tfidf_weights.pkl')
    X = vectorizer.transform(text)
    return X

# Function to predict using the model
def predict_on_model(text):
    # Load the Random Forest model
    model = load('./weights/RandomForestClassifier_fold_5_weights.pkl')

    X = vectorize_text(text)

    # Make predictions
    return model.predict(X)

# Function to get the label
def get_label(prediction):
    labels = ['adulting-101','big-read','commentary','gen-y-speaks','gen-z-speaks','singapore','voices','world']
    flat_prediction = np.asarray(prediction).flatten()  # Convert to 1D array
    label_indices = np.nonzero(flat_prediction)[0]  # Find non-zero indices
    return [labels[i] for i in label_indices]


# Test the function
text = ['app new feature serve convenience value whether looking dine take noticed price creeping wallet taking hit transport daily essential dining cost living steadily rising leaving consumer business feeling pinch food significant household expenditure singapore grab taking proactive approach ease financial burden without compromising food craving includes reducing delivery fee presenting value money dine deal aimed making daily life bit affordable everyday superapp important find way create lower priced service consumer want prudent spending still enjoying convenience demand service said mr tay chuen jein head delivery grab singapore whether gearing night opting cosy evening home five way grabfood making sure budget go distance 1 smart meal scheduling savvy saving thanks grab saver delivery option offer reduced fee exchange slightly extended delivery timeframe unlock saving ordering meal advance best part still savour meal exactly want smart planning made possible process called hyper batching platform leverage artificial intelligence identify group order sequence multiple delivery along overlapping route taking account real time factor traffic weather condition technology allows u assign order similar delivery route one delivery partner mean multiple consumer sharing cost delivery get pay little le mr tay explained nifty feature saved foodie singapore u 7 million 9 4 million delivery fee eight month time delivery partner boost earnings completing order le time effort making win win situation everyone 2 merrier cheaper communal gathering become regular part day day life grabfood updated group order feature gamechanger enabling user enjoy convenience cost saving participating group order breeze simply send invitation link scan qr code add order shared cart group order also used mix match feature let consumer order variety merchant located similar area way something everyone group even settling bill afterwards effortlessly managed app smart bill calculation feature automatically computes person share ensuring fair split includes delivery fee tax tip evenly distributed among participant according mr tay group order especially popular snack like bubble tea meeting minimum spend often requires pooling several order function also help reduce hassle involved consolidating order allow customisation adjusting sugar level topping bubble tea added 3 dine without splash city bursting culinary option hard decide dine fortunately bargain seeker look grabfood dine feature inspiration treasure trove exclusive restaurant perk designed help user explore new gastronomic experience without denting wallet tap browse nearby deal read restaurant review pre purchase dine voucher even book ride chosen restaurant imagine satisfying sushi craving maki san generous 30 per cent discount cash voucher indulging delectable chinese new year set soup restaurant 30 per cent discount said mr tay work merchant partner create value money meal deal allowing consumer save enjoying restaurant dining experience notably 35 per cent dine user either never used grabfood done past three month show reaching previously underserved group 4 unlock saving subscription plan grabfood staple meal routine grabunlimited perfect choice cost conscious convenience loving singaporean monthly fee 5 99 subscription plan open door exclusive deal spanning across 1 000 merchant app perfect family dinner social gathering large food order come added perk 10 delivery fee beyond realm food grabunlimited let subscriber enjoy 10 per cent grabcar premium even quick grocery run via grabmart economical 3 delivery fee 5 browse best deal one roof offer user even bang buck grabfood recently unveiled meal 7 landing page app easily discover grabfood best deal complete customisable filter price point delivery budget well current promotion help make feature grab continues grow focus affordability aligns mission drive southeast asia forward making service accessible local community offering create value consumer use grab everyday life continue find unique way better serve foodie singapore sustainable manner said mr tay irresistible food deal download grab app today']
prediction = predict_on_model(text)
label = get_label(prediction[0])
print(label)
print(prediction)
