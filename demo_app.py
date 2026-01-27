import streamlit as st
import plotly.graph_objects as go
import requests
import json
import base64
from PIL import Image
import datetime

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Skin Analysis AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Classes for Object Access ---
class DictObj:
    def __init__(self, in_dict: dict):
        if not isinstance(in_dict, dict):
            self._value = in_dict
            return

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
    
    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' not found")

# --- Helper Functions ---
def create_spider_chart(analysis):
    categories = ['Moisture', 'Oiliness', 'Wrinkles', 'Spots', 'Pores', 'Redness', 'Acne', 'Dark Circles']
    # Safely access nested attributes using getattr chaining or checks
    def get_score(parent, attr):
        obj = getattr(parent, attr, None)
        return getattr(obj, 'score', 0) if obj else 0

    values = [
        get_score(analysis, 'moisture'),
        get_score(analysis, 'oiliness'),
        get_score(analysis, 'wrinkles'),
        get_score(analysis, 'spots'),
        get_score(analysis, 'pores'),
        get_score(analysis, 'redness'),
        get_score(analysis, 'acne'),
        get_score(analysis, 'dark_circles'),
    ]
    
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='rgb(99, 102, 241)', width=3),
        name='Skin Analysis'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, mode='markers',
        marker=dict(
            size=12, color=values,
            colorscale=[[0, 'rgb(239, 68, 68)'], [0.5, 'rgb(251, 146, 60)'], [1, 'rgb(34, 197, 94)']],
            cmin=0, cmax=100, line=dict(color='white', width=2), showscale=False
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            angularaxis=dict(tickfont=dict(size=12, color='#374151')),
            bgcolor='rgba(249, 250, 251, 0.8)'
        ),
        margin=dict(l=80, r=80, t=40, b=40),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif')
    )
    return fig

# --- Utility: compute age from date of birth ---
def compute_age(dob: datetime.date, today: datetime.date = None) -> int:
    if not dob:
        return None
    if today is None:
        today = datetime.date.today()
    years = today.year - dob.year
    # subtract one if birthday hasn't occurred yet this year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years

# --- UI Layout ---

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=50)
    st.title("⚙️ Controls")
    
    # API Configuration
    st.subheader("API Connection")
    api_ip = st.text_input("API IP Address", value="13.219.77.10")
    api_port = st.text_input("API Port", value="8000")
    
    base_url = f"http://{api_ip}:{api_port}"

    # Ask user for their Date of Birth (required)
    st.subheader("User Information")
    # Use a text input with explicit format so there's no pre-filled default value.
    dob_input = st.text_input("Date of Birth (required) — YYYY-MM-DD", value="", placeholder="e.g. 1996-04-23")
    dob = None
    actual_age = None
    valid_dob = False

    if dob_input:
        try:
            dob_parsed = datetime.date.fromisoformat(dob_input.strip())
            if dob_parsed > datetime.date.today():
                st.error("Date of birth cannot be in the future.")
            else:
                dob = dob_parsed
                actual_age = compute_age(dob)
                valid_dob = True
        except Exception:
            st.error("Invalid date format. Please enter in YYYY-MM-DD.")
    else:
        st.warning("Please enter your date of birth (required) in the sidebar before uploading an image.")

    if st.button("Check Connection"):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error(f"❌ Error: {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Failed: {e}")

    st.divider()
    show_json = st.toggle("Show Raw JSON Output", value=False)
    
    st.caption("Powered by Taiuo Skin Analysis API")

# Main Header
st.title("🧬 AI Skin Health Analysis")
st.markdown("Upload a face image to generate a comprehensive dermatological assessment.")

# File Uploader
# File uploader is shown only after a valid DOB is provided
uploaded_file = None
if 'valid_dob' in globals() and valid_dob:
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'], label_visibility="collapsed")
else:
    st.info("Please provide your date of birth in the sidebar first. The image uploader will appear after you enter a valid DOB.")

if uploaded_file:
    # Use PIL to load the image robustly
    try:
        pil_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Create layout columns
    col_img, col_trace = st.columns([1, 1])

    # 1. Image Visualization Column
    with col_img:
        st.subheader("📸 Input Analysis")
        # Simply display the PIL image. This avoids any numpy/cv2 complications.
        st.image(pil_image, caption="Original Image", width='stretch')

    # 2. Pipeline Execution Column (Using API)
    with col_trace:
        st.subheader("📍 Analysis Status")
        
        # Define placeholders for state
        result = None
        raw_data = None
        
        # Auto-start analysis
        with st.status("Running Remote Analysis...", expanded=True) as status:
            st.write("📤 Uploading image to API...")
            try:
                # Prepare file for upload
                uploaded_file.seek(0)
                
                files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                analyze_url = f"{base_url}/analyze"
                
                # Make API Call
                st.write("🔄 Processing on GPU Cluster...")
                response = requests.post(analyze_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    st.write("✅ Analysis Complete!")
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    
                    # Convert dict to object for dot notation access to match original UI code
                    raw_data = data
                    result = DictObj(data)
                    
                else:
                    st.error(f"API Error: {response.text}")
                    status.update(label="Analysis Failed", state="error")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
                status.update(label="Connection Failed", state="error")

    # 3. Results Section
    if result and getattr(result, 'status', '') == 'success':
        st.divider()
        st.subheader("📊 Assessment Report")
        
        # Score Cards
        score = getattr(result, 'evaluation_score', 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Health Score", f"{score}/100", delta_color="normal")
        
        rating = "Excellent" if score >= 76 else "Good" if score >= 51 else "Fair" if score >= 26 else "Poor"
        c2.metric("Rating", rating)
        
        # Access nested objects
        skin_analysis = getattr(result, 'skin_analysis', None)
        concerns_count = len(getattr(skin_analysis, 'top_concerns', [])) if skin_analysis else 0
        c3.metric("Concerns Identified", concerns_count, delta_color="inverse")

        # Spider Web Chart Section
        if skin_analysis:
            st.markdown("---")
            st.markdown("#### 🕸️ Skin Analysis Overview")
            
            chart_col, info_col = st.columns([3, 2])
            
            with chart_col:
                spider_fig = create_spider_chart(skin_analysis)
                st.plotly_chart(spider_fig, width='stretch')
            
            with info_col:
                st.markdown("##### 📌 Key Insights")
                st.markdown("""
                The spider chart provides a visual overview of your skin health across 8 key parameters:
                - **Higher scores** (outer edges) = Better condition
                - **Lower scores** (inner area) = Needs attention
                """)
                
                if getattr(skin_analysis, 'top_concerns', []):
                    st.markdown("##### ⚠️ Top Concerns")
                    for concern in skin_analysis.top_concerns[:3]:
                        param_name = getattr(concern, 'parameter', 'Unknown')
                        st.warning(f"• {param_name}")

        # Detailed Parameters Grid
        st.markdown("#### Parameter Breakdown")
        
        if skin_analysis:
            def param_row(label, feature_obj):
                if not feature_obj: return
                cols = st.columns([2, 4, 1])
                with cols[0]:
                    st.markdown(f"**{label}**")
                    severity = getattr(feature_obj, 'severity', 'unknown')
                    st.caption(severity.capitalize())
                with cols[1]:
                    score = getattr(feature_obj, 'score', 0)
                    st.progress(score / 100)
                with cols[2]:
                    st.write(f"{getattr(feature_obj, 'score', 0)}/100")

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                param_row("💧 Moisture", getattr(skin_analysis, 'moisture', None))
                param_row("🫧 Oiliness", getattr(skin_analysis, 'oiliness', None))
                param_row("〰️ Wrinkles", getattr(skin_analysis, 'wrinkles', None))
                param_row("🔵 Spots", getattr(skin_analysis, 'spots', None))
            with p_col2:
                param_row("⚫ Pores", getattr(skin_analysis, 'pores', None))
                param_row("🔴 Redness", getattr(skin_analysis, 'redness', None))
                param_row("💢 Acne", getattr(skin_analysis, 'acne', None))
                param_row("👁️ Dark Circles", getattr(skin_analysis, 'dark_circles', None))

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            rec = getattr(skin_analysis, 'recommendations', None)
            if rec:
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.markdown("##### Immediate Actions")
                    for action in getattr(rec, 'immediate_actions', []):
                        cat = getattr(action, 'product_category', '')
                        sug = getattr(action, 'suggestion', '')
                        st.info(f"**{cat}**: {sug}")     
                with r_col2:
                    st.markdown("##### Maintenance Tips")
                    for tip in getattr(rec, 'maintenance_tips', []):
                        st.success(f"• {tip}")
        
        # Product Recommendations
        st.markdown("---")
        st.subheader("🛒 Personalized Product Recommendations")
        
        # --- Debug Section ---
        with st.expander("🕵️ Product Recommendation Debugger", expanded=True):
            st.write("**Step 1: Checking Data Source**")
            if raw_data:
                st.success("✅ `raw_data` is available.")
            else:
                st.error("❌ `raw_data` is missing.")

            st.write("**Step 2: Extracting Recommendation Block**")
            product_recommendations_data = raw_data.get("product_recommendations") if raw_data else None
            
            if product_recommendations_data:
                 st.success("✅ `product_recommendations` key found.")
                 st.json(list(product_recommendations_data.keys()))
            else:
                 st.warning("⚠️ `product_recommendations` key NOT found in API response.")
                 st.write("Available keys:", list(raw_data.keys()) if raw_data else "None")

        # --- Logic Section ---
        # Using raw_data (formerly trace)
        if product_recommendations_data:
            st.markdown("""
            Based on your skin analysis, we've curated the **best products** from our catalog
            that directly address your top skin concerns.
            """)
            
            recommendation_result = product_recommendations_data
            
            # Display Top Concerns Identified
            top_concerns = recommendation_result.get("top_concerns_identified", [])
            
            st.write(f"**Debug Step 3: Top Concerns Count:** {len(top_concerns)}")
            
            if top_concerns:
                st.markdown("##### ⚠️ Top Concerns Identified")
                concern_cols = st.columns(len(top_concerns))
                for i, concern in enumerate(top_concerns):
                    with concern_cols[i]:
                        severity = concern.get("severity", "mild")
                        severity_color = "🔴" if severity == "severe" else "🟠" if severity == "moderate" else "🟡"
                        st.metric(
                            label=f"{severity_color} {concern.get('parameter', 'Unknown').replace('_', ' ').title()}",
                            value=f"{concern.get('score', 0)}/100",
                            delta=f"-{concern.get('improvement_potential', 0)}% potential",
                            delta_color="inverse"
                        )
            
            st.markdown("")
            st.markdown("##### 🎁 Recommended Products")
            
            recommended_products = recommendation_result.get("recommended_products", [])
            st.write(f"**Debug Step 4: Product Count:** {len(recommended_products)}")
            
            if recommended_products:
                prod_cols = st.columns(4)
                tier_badges = {
                    "drugstore": ("💰", "#22c55e", "Drugstore"),
                    "mid": ("💎", "#3b82f6", "Mid-Range"),
                    "premium": ("👑", "#a855f7", "Premium")
                }
                
                for i, product in enumerate(recommended_products[:4]):
                    with prod_cols[i]:
                        price_tier = product.get('price_tier', 'mid')
                        tier_info = tier_badges.get(str(price_tier), ("💎", "#3b82f6", "Mid-Range"))
                        tier_icon, tier_color, tier_label = tier_info
                        
                        category = product.get('category', 'Product')
                        name = product.get('name', 'Product')
                        brand = product.get('brand', 'Brand')
                        
                        # Card container with styling
                        with st.container():
                            badge_html = f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="background: {tier_color}20; color: {tier_color}; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;">{tier_icon} {tier_label}</span>
                                <span style="background: #f1f5f9; color: #475569; padding: 4px 8px; border-radius: 8px; font-size: 11px;">{category.title()}</span>
                            </div>
                            """
                            st.markdown(badge_html, unsafe_allow_html=True)
                            st.markdown(f"**{name}**")
                            st.caption(f"by {brand}")
                            
                            # Active Ingredients
                            actives = product.get('key_actives', [])
                            if actives:
                                actives_str = ", ".join([a.replace('_', ' ').capitalize() for a in actives[:3]])
                                st.markdown(f"<small>🧪 <b>Actives:</b> {actives_str}</small>", unsafe_allow_html=True)

                            # Concerns
                            concerns = product.get('concerns_addressed', [])
                            if concerns:
                                # Ensure concerns are strings
                                c_list = [c if isinstance(c, str) else c.get('parameter', '') for c in concerns]
                                concern_tags = " ".join([f"`{c}`" for c in c_list])
                                st.markdown(f"<small>🎯 <b>Targets:</b> {concern_tags}</small>", unsafe_allow_html=True)
                            
                            # Notes/Relevance
                            notes = product.get('notes', '')
                            if notes:
                                st.info(f"{notes}")

                            buy_url = product.get('buy_url', '#')
                            if buy_url and buy_url != '#':
                                st.link_button("View Product →", buy_url, use_container_width=True)
                            else:
                                st.button("View Product →", key=f"prod_{i}", disabled=True, use_container_width=True)
                            st.markdown("---")
            else:
                st.info("No curated products found for your specific profile.")
                
                st.markdown("")
                
                # Show JSON toggle for recommendations
                with st.expander("📄 View Recommendation JSON"):
                    st.json(recommendation_result)

        # AWS Rekognition
        aws_data = getattr(result, 'aws_rekognition', None)
        if aws_data:
            st.markdown("---")
            st.subheader("👁️ AWS Rekognition - Face Analysis")
            aws_col1, aws_col2, aws_col3 = st.columns(3)
            
            with aws_col1:
                age = getattr(aws_data, 'age_range', None)
                if age:
                    # Adjust AWS predicted age range by averaging with the user's provided DOB-derived age.
                    aws_low = getattr(age, 'low', None)
                    aws_high = getattr(age, 'high', None)

                    try:
                        if aws_low is not None and actual_age is not None:
                            adj_low = round((float(aws_low) + float(actual_age)) / 2)
                        else:
                            adj_low = aws_low if aws_low is not None else '?'

                        if aws_high is not None and actual_age is not None:
                            adj_high = round((float(aws_high) + float(actual_age)) / 2)
                        else:
                            adj_high = aws_high if aws_high is not None else '?'
                    except Exception:
                        adj_low, adj_high = aws_low, aws_high

                    st.metric("🎂 Estimated Age", f"{adj_low}-{adj_high} years")
                
                gender = getattr(aws_data, 'gender', None)
                if gender:
                    st.metric("👤 Gender", f"{getattr(gender, 'value', 'Unknown')}")
            
            with aws_col2:
                emotion = getattr(aws_data, 'primary_emotion', None)
                if emotion:
                    st.metric("😊 Primary Emotion", getattr(emotion, 'type', 'Unknown'))
                
                quality = getattr(aws_data, 'quality', None)
                if quality:
                    st.metric("☀️ Brightness", f"{getattr(quality, 'brightness', 0):.1f}%")

            with aws_col3:
                st.markdown("**Facial Attributes**")
                attrs = [("smile", "😄 Smile"), ("eyeglasses", "👓 Glasses"), ("beard", "🧔 Beard"), ("eyes_open", "👁️ Eyes Open")]
                for key, label in attrs:
                    attr = getattr(aws_data, key, None)
                    if attr:
                        val = getattr(attr, 'value', False)
                        icon = "✅" if val else "❌"
                        st.write(f"{icon} {label}")

        # Skin Tone
        skin_tone_data = getattr(result, 'skin_tone', None)
        if skin_tone_data:
            st.markdown("---")
            st.subheader("🍞 Skin Tone Classification")
            summary = getattr(skin_tone_data, 'summary', '')
            if summary: st.success(f"✅ {summary}")
            
            faces = getattr(skin_tone_data, 'faces', [])
            tone_names = {
                "AI": "Very Light (Type I)", "AII": "Light (Type II)", "AIII": "Medium Light (Type III)",
                "BI": "Medium (Type IV)", "BII": "Medium Dark (Type V)", "BIII": "Dark (Type VI)"
            }

            if faces:
                if isinstance(faces, list): faces_list = faces
                else: faces_list = faces # DictObj wrapper behaves iterable

                for i, face in enumerate(faces_list):
                    st.markdown(f"**👤 Face {i + 1}**")
                    tone_col1, tone_col2, tone_col3 = st.columns(3)
                    with tone_col1:
                        skin_tone = getattr(face, 'skin_tone', '#000000')
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="width: 50px; height: 50px; background-color: {skin_tone}; border-radius: 8px; border: 2px solid #ccc;"></div>
                            <div><strong>Skin Tone</strong><br/><code>{skin_tone}</code></div>
                        </div>""", unsafe_allow_html=True)
                    with tone_col2:
                        tone_label = getattr(face, 'tone_label', 'Unknown')
                        st.metric("🏷️ Classification", tone_names.get(tone_label, tone_label))
                        st.metric("🎯 Accuracy", f"{float(getattr(face, 'accuracy', 0)):.2f}%")
                    with tone_col3:
                        dominant_colors = getattr(face, 'dominant_colors', [])
                        if dominant_colors:
                            st.markdown("**🌈 Dominant Colors**")
                            for color in dominant_colors:
                                hex_color = getattr(color, 'color', '#000')
                                percent = getattr(color, 'percent', 0)
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                    <div style="width: 20px; height: 20px; background-color: {hex_color}; border-radius: 4px; border: 1px solid #ccc;"></div>
                                    <span><code>{hex_color}</code> ({float(percent) * 100:.1f}%)</span>
                                </div>""", unsafe_allow_html=True)

        # Disease Detection
        disease_data = getattr(result, 'disease_detection', None)
        if disease_data:
            st.markdown("---")
            st.subheader("🦠 Disease Detection")
            raw_result = getattr(disease_data, 'raw_result', None)
            if raw_result:
                pred_class = getattr(raw_result, 'predicted_class', 'Unknown')
                conf = getattr(raw_result, 'confidence', 0)
                st.metric("Primary Prediction", pred_class.replace('_', ' ').title(), f"{conf*100:.1f}% confidence")
                st.warning("⚠️ Disclaimer: Not a medical diagnosis.")

        # Heatmap
        heatmap_b64 = getattr(result, 'heatmap_base64', None)
        if heatmap_b64:
            st.markdown("---")
            st.subheader("🔥 Skin Analysis Heatmap")
            try:
                heatmap_bytes = base64.b64decode(heatmap_b64)
                st.image(heatmap_bytes, caption="AI Generated Heatmap", width='stretch')
            except Exception as e:
                st.error("Error displaying heatmap.")

    elif result:
         st.error(f"Analysis Failed: {getattr(result, 'error_message', 'Unknown Error')}")

    # JSON output
    if result:
        st.divider()
        st.subheader("📄 Raw Data")
        
        # Debugging
        st.write("Debug Info:")
        if raw_data:
            st.write("Top-level keys:", list(raw_data.keys()))
            if 'product_recommendations' in raw_data:
                st.write("Product Reco Keys:", list(raw_data['product_recommendations'].keys()))
            else:
                st.warning("No 'product_recommendations' key found in API response.")

        with st.expander("View Full API Response JSON", expanded=True):
             st.json(raw_data)

else:
    # Empty State
    st.info("👆 Please upload an image to begin the demo.")
