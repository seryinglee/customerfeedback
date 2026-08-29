import streamlit as st
import plotly.express as px
from model import evaluate_models

st.set_page_config(page_title="Model Comparison", page_icon="📊")

st.title("📊 Model Performance Comparison")
st.warning("Model evaluation trains all models and may take some time.")

if st.button("Run Model Evaluation", type="primary"):
    try:
        with st.spinner("Training and evaluating models..."):
            df_eval = evaluate_models()

        if not df_eval.empty:
            st.dataframe(df_eval.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }), use_container_width=True)

            df_melted = df_eval.melt(id_vars="Model", var_name="Metric", value_name="Score")

            st.markdown("#### 📊 Bar Chart")
            fig_bar = px.bar(
                df_melted,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text_auto=".1%",
                height=450,
            )
            fig_bar.update_layout(yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.warning("Evaluation could not be performed.")
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
else:
    st.info("Click the button above to compare model performance")
